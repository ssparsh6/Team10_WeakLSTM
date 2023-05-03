"""
04/17/2023
    Training Script for training a Weakly supervised NN: (custom untrained) ResNet18 + ConvLSTM + Conv1x1
    Hyperparameters/Specs: 
        1. LR: constant LR 10e-5
        2. Batch: 4, Seq: 4
        3. Opt: Adam
        4. Loss: Weighted BCE
        5. Dataset: Full CholecT50
        6. Epochs: 150
        7. Early Stopping: Yes
        8. Checkpointing: Yes, path: WeakLSTM/trainingData_0417/checkpoints
        
04/18/2023:
    - Model overfit
    - kernel_regularizer is added to ResNet18 (1e-5)
    - Dropout added to first layer (0.4)
    - Added LR Scheduler. Starting at 1e-4
    
04/19/2023:
    - Added Norm Preprocessing - uses 2000 "batches" to find mean and variance
    
04/20/2023:
    - Removed Norm and reverted model back to 04/17/2023 except one change - updated convlstm kernel to 5,5 from 1,1
        
"""

# Import packages
import tensorflow as tf
import os, sys
import warnings
from cholect50 import dataloader_tf as dataloader
import numpy as np
from tensorflow.nn import weighted_cross_entropy_with_logits as loss_fn
import datetime
import tensorflow_models as tfm
from tensorflow_models import vision
from keras import layers
from keras.layers import Input, Activation, ZeroPadding2D, BatchNormalization, Conv2D, MaxPooling2D, ConvLSTM2D, TimeDistributed, Dropout
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras import regularizers

from keras.layers import Layer
import tensorflow_addons as tfa

version = "Final/"
ModelPathLocation = "WeakLSTM/"

# Paths 
savedModelPath = ModelPathLocation + "finalModel/" + version
ckptPath = ModelPathLocation + "checkpoints/" + version
logsPath = ModelPathLocation + "logs/" + version

# Configure GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# Load dataset

BATCH_SIZE = 4
SEQUENCE_DEPTH = 5

dataset = dataloader.CholecT50( 
          dataset_dir="CholecT50/", 
          dataset_variant="cholect50",
          test_fold=1,
          augmentation_list=['original'],
          num_parallel_calls=100,
          num_sequence = SEQUENCE_DEPTH
          )

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()

# train and val data loaders
train_dataloader = train_dataset.batch(BATCH_SIZE) # see tf.data.Dataset for more options
val_dataloader   = val_dataset.batch(BATCH_SIZE)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = video_dataset.batch(BATCH_SIZE)
    test_dataloaders.append(test_dataloader)  
    
train_dataloader = train_dataloader.prefetch(tf.data.AUTOTUNE)
val_dataloader = val_dataloader.prefetch(tf.data.AUTOTUNE)

# Given instruments in the dataset
instruments = np.array([
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator"
])

# Take 1 batch of image sequences from the train_dataloader
for (img_list, (ivt_list, i_list, v_list, t_list, p_list)) in train_dataloader.take(1):
    break
    
print("Image List Shape: ", img_list.shape)

# Normalization
#norm_layer = layers.Normalization()
#norm_layer.adapt(data=train_dataloader.take(1000).map(map_func=lambda img, labels: img))

#print("Done Normalization")

# Model Dev
def wildcat_pooling(img, alpha=0.6, name='Wildcat_Pooling'):
    # Axis: Breadth and Width of the input tensor. Assuming
    # 0 is the batch size. Check if we have a 5D Tensor
    with tf.name_scope(name):
        return tf.math.reduce_max(img, axis=[-3,-2]) + alpha*tf.math.reduce_min(img, axis=[-3,-2])
    
# Define the input as a tensor with shape input_shape
filters = [64, 128, 256, 512]
strides = [1,   2,   1,   1]

X_input = Input(shape = (256, 448, 3))

# Zero-Padding
X = ZeroPadding2D((3, 3))(X_input)

# Stage 1
X = Conv2D(64, (7, 7), 
           strides=(2, 2), 
           name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(name='bnConv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)


# Stage 2
X = vision.layers.ResidualBlock(filters[0], strides[0], 
                                use_projection=False)(X)
X = vision.layers.ResidualBlock(filters[0], strides[0], 
                                use_projection=False)(X)

# Stage 3
X = vision.layers.ResidualBlock(filters[1], strides[1], 
                                use_projection=True)(X) #First
X = vision.layers.ResidualBlock(filters[1], strides[0], 
                                use_projection=False)(X)

# Stage 4
X = vision.layers.ResidualBlock(filters[2], strides[2], 
                                use_projection=True)(X) #First
X = vision.layers.ResidualBlock(filters[2], strides[0], 
                                use_projection=False)(X)

# Stage 5
# X = vision.layers.ResidualBlock(filters[3], strides[3], 
#                                 use_projection=True)(X) #First
# X = vision.layers.ResidualBlock(filters[3], strides[0], 
#                                 use_projection=False)(X)

resnetModel = Model(inputs=X_input, outputs=X, name='resnet-custom')

X_input = Input(shape = (SEQUENCE_DEPTH, 256, 448, 3))
reshaped_input = TimeDistributed(resnetModel)(X_input)

X = ConvLSTM2D(
     filters=6,
     kernel_size=(5, 5), 
     name='convLSTMLayer', 
     padding = 'same',
     kernel_regularizer = regularizers.L2(1e-5),
     return_sequences=True)(reshaped_input)

LHMap = Conv2D(filters=6, kernel_size=(1, 1), name='LocMapLayer')(X)
X = wildcat_pooling(LHMap)

model = Model(inputs=X_input, outputs=[X, LHMap], name='WNet')

print(model.summary())

# Set up for Training 
class_weights = tf.convert_to_tensor(
    [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455], 
    dtype=tf.float32, dtype_hint=None, name=None
)

# Define the learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        #return max(lr * tf.math.exp(-0.3),1e-6)
        return 1e-6
    
    
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer
)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# TF Summary Writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = logsPath + current_time + '/train'
val_log_dir = logsPath + current_time + '/val'
lr_log_dir = logsPath + current_time + '/lr'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)
lr_summary_writer = tf.summary.create_file_writer(lr_log_dir)

def train_step(model, optimizer, inputs, targets):
    
    with tf.GradientTape() as tape:
    
        [logits, LHMap] = model(inputs, training=True)  # Logits for this minibatch
        
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(labels=targets, logits=logits, pos_weight=class_weights)
        loss_value = tf.math.reduce_mean(loss_value)
        
    gradients = tape.gradient(loss_value, model.trainable_variables)
    
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss_value

def validation_step(model, inputs, targets):
    [logits, LHMap] = model(inputs, training=False)  # Logits for this minibatch
    
    # Compute the loss value for this minibatch.
    loss_value = loss_fn(labels=targets, logits=logits, pos_weight=class_weights)
    loss_value = tf.math.reduce_mean(loss_value)
        
    return loss_value



# Training Setup
# Training Parameters
num_training_samples = 72024 #72024 #13306
epochs = 150
everyNEpochs = 5
patience = 5
wait = 0
best = float('inf')

# Metrics
metrics_names = ['train_loss', 'val_loss']
train_loss_results = []
val_loss_results = []
epoch_loss_avg = tf.keras.metrics.Mean()
val_epoch_loss_avg = tf.keras.metrics.Mean()

for epoch in range(epochs):  
    
    print("\nepoch {}/{}".format(epoch+1,epochs))
    progBar = tf.keras.utils.Progbar(num_training_samples, stateful_metrics=metrics_names)
    
    # Training
    for step, (img, (_, label_i, _, _, _)) in enumerate(train_dataloader):      
        loss_value = train_step(model, optimizer, img, label_i)
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                
        values=[('train_loss',epoch_loss_avg.result())]
        progBar.update(step*BATCH_SIZE*SEQUENCE_DEPTH, values=values)
    
    #print("Steps: ", 20*step)
    
    # Log Training Loss
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
        
    
    # Validation
    for step, (img, (_, label_i, _, _, _)) in enumerate(val_dataloader):
        val_loss_value = validation_step(model, img, label_i)
        val_epoch_loss_avg.update_state(val_loss_value)
        
    # Log Validation Loss
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_epoch_loss_avg.result(), step=epoch)
        
    # Update Progress bar
    values=[('train_loss',epoch_loss_avg.result()),
            ('val_loss',val_epoch_loss_avg.result())
            ]
    progBar.update(num_training_samples, values=values, finalize=True)         

    # Append to results
    train_loss_results.append(epoch_loss_avg.result())
    val_loss_results.append(val_epoch_loss_avg.result())
    
    
    # Update learning rate at the end of each epoch
    lr = optimizer.learning_rate.numpy()
    new_lr = lr_scheduler(epoch, lr)
    optimizer.learning_rate.assign(new_lr)
    
    print("Epoch {:03d}: Train Loss: {:.3f}, Val Loss: {:.3f}, Learning Rate: {:.6f}".format(
        epoch+1,
        epoch_loss_avg.result(),
        val_epoch_loss_avg.result(),
        lr
        )
    )
        
    # Log Training Loss
    with lr_summary_writer.as_default():
        tf.summary.scalar('LR', lr, step=epoch)
    
    # Save Checkpoint every 5 epochs
    if epoch % everyNEpochs == 0:
        savedTo = checkpoint.save(ckptPath + "epoch")
        print("Saved checkpoint for step {}: {}".format(int(epoch), savedTo))
        
    # Early Stoppping
    wait += 1
    if val_epoch_loss_avg.result() < best:
        best = val_epoch_loss_avg.result()
        wait = 0
    if wait >= patience:
        print("Stopping Early", "Epoch Number: ", epoch +1)
        break
    
    # Reset losses after every epoch
    epoch_loss_avg.reset_states()
    val_epoch_loss_avg.reset_states()
    
    
# Save tensorflow recommened way
model.save(savedModelPath)


