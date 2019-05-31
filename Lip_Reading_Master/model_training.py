#!/usr/bin/env python
# coding: utf-8

# ## LOAD DATA AND ASSIGN LABELS ##

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense, Convolution3D, ZeroPadding3D, Activation, MaxPooling3D, Flatten, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model

np.random.seed(7)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


# In[30]:


# source folder of video files
video_path = 'path_of_video_source'
# destination folder of preprocessed samples
NPY_FOLDER = os.path.join("NPY", "npy_28")

# sizes of mouth region -> input shape
WIDTH = 24
HEIGHT = 32
DEPTH = 28

# print info when processing data
debug = False

# list all words
print(", ".join(os.listdir(video_path)))


# In[31]:


# train_words subset of words for training
train_words = ["ABOUT", "ACCESS", "ALLOW", "BANKS", "BLACK", "CALLED", "CONCERNS",
            "CRISIS", "DEGREES", "DIFFERENT", "DOING", "EDITOR", "ELECTION",
            "EVERY", "FOCUS", "GROUP", "HUMAN", "IMPACT", "JUSTICE"]


# In[32]:


classes = len(train_words)  # len(words)

labels = train_words
num_labels = [i for i in range(0, len(labels))]
hot_labels = to_categorical(num_labels)

num_labels_dict = dict(zip(labels, num_labels))
hot_labels_dict = dict(zip(labels, hot_labels))


# In[35]:


def augment_data(sample):
    if np.random.rand() > .5: sample = np.flip(sample, 2) # flip the sample horizontally
    return sample


# In[36]:


def sample_generator(basedir, set_type, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    
    # Directory from which to load samples
    directory = os.path.join(basedir, set_type)
    
    # Create NumPy arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, DEPTH,WIDTH, HEIGHT, 1))
    batch_labels = np.zeros((batch_size, classes), dtype="uint8")
    
    file_list = []
    # Populate with file paths and labels
    for word_folder in labels:
        file_list.extend((word_folder, os.path.join(directory, word_folder, word_name))                          for word_name in os.listdir(os.path.join(directory, word_folder)))
    
    while True:
        for b in range(batch_size):
            
            i = np.random.choice(len(file_list), 1)[0]
            
            sample = np.load(file_list[i][1])  # get random sample
            
            # Normalize to [-1; 1]
            sample = (sample.astype("float16") - 128) / 128 
            sample = sample.reshape(sample.shape + (1,))
    
            batch_features[b] = sample
            batch_labels[b] = hot_labels_dict[file_list[i][0]]  # get hot_labels vector

        yield (batch_features, batch_labels)


# ## Shape and data type of generated samples 

# In[37]:


a = sample_generator(NPY_FOLDER, "train", 2)
s = next(a)
print(s[0].dtype, s[1].dtype)
print(s[0].shape, s[1].shape)


# In[38]:


# load all data at once into arrya - e.g when evaluating the test subset
def load_data(basedir, set_type):

    directory = os.path.join(basedir, set_type)
    file_list = []
    for word_folder in labels:
        file_list.extend((word_folder, os.path.join(directory, word_folder, word_name))                          for word_name in os.listdir(os.path.join(directory, word_folder)))
        
    shuffle(file_list)
    
    X = []
    y = []
    
    for f in file_list:
        
        sample = np.load(f[1])
        sample = (sample.astype("float16") - 128) / 128  # normalize to 0 - 1
        X.append(sample)
        
        y.append(labels_hot_dict[f[0]])
    
    X = np.array(X)
    X = X.reshape(X.shape + (1,))

    return (X, np.array(y))


# ## Train the Lip Reading Model ##

# In[39]:


models_dir = 'lip_reading_models/'
model_name = "model_D"


# In[44]:


def get_model(model_architecture, dropout_rate):
    
    
    if model_architecture == "EF-3":
        model = Sequential()
        model.add(Convolution3D(48, (3, 3, 3), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Convolution3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        
    elif model_architecture == "model_A1":
        model = Sequential()
        model.add(Convolution3D(64, (3, 3, 3), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Convolution3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(Convolution3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
    
    elif model_architecture == "model_A2":
        model = Sequential()
        model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        
    elif model_architecture == "model_B":
        model = Sequential()
        model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        
    elif model_architecture == "model_C":
        model = Sequential()
        model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Dropout(dropout_rate))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        
    elif model_architecture == "model_D":
        model = Sequential()
        model.add(Convolution3D(64, (5, 5, 5), padding='same', activation='relu', input_shape=(DEPTH, WIDTH, HEIGHT, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Dropout(dropout_rate))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(Convolution3D(128, (5, 5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(3, 3, 3)))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))
        
    return model
    


# ## Pass the Adam optimzer and compile model

# In[45]:


opt = Adam(lr=1e-4)

model = get_model("model_D", 0.4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# ## Save and plot the 3D-CNN architecture 

# In[46]:


# save model architecture to json
model_json = model.to_json()
with open(models_dir + model_name + ".json", "w") as json_file:
    json_file.write(model_json)

# plot the model architecture
model.summary()
plot_model(model, to_file='outputs/architecture_{}.pdf'.format(model_name), show_shapes=True, show_layer_names=False)


# ## Initialize Keras Callback

# In[48]:


tensorboard = TensorBoard(log_dir="logs/{}".format(model_name),
                          write_graph=True, write_images=True)
    
filepath = models_dir + model_name + ".h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='max')

earlyStopping = EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')

csv_logger = CSVLogger('outputs/log_{}.csv'.format(model_name), append=True, separator=';')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.0001)


# ## Run the training model

# In[47]:


nb_epoch = 30
batch_size = 16

num_examples = classes * 900
num_val_examples = classes * 90

history = model.fit_generator(
          generator=sample_generator(NPY_FOLDER, "train", batch_size),
          epochs=nb_epoch,
          steps_per_epoch=num_examples // batch_size,
          validation_data=sample_generator(NPY_FOLDER, "val", batch_size),
          validation_steps=num_val_examples // batch_size,
          verbose=True,
          callbacks = [tensorboard, checkpoint, earlyStopping, csv_logger] # learning_rate_reduction
)


# ## Plot and Save the training results

# In[49]:


def plot_and_save_training():
    plt.figure(1, figsize=(8,8))
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('no. of epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('no. of epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    
    plt.savefig('outputs/train_{}.pdf'.format(model_name))
    plt.show()

plot_and_save_training()


# In[ ]:




