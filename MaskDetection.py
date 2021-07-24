#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np  


# In[2]:


IMG_SIZE = 150

# Parametros data augmentation
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')


# In[3]:


# Dataset de treino

TRAINING_DIR = "E:/MaskImages/Train"
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='categorical',
    batch_size=32
)


# In[4]:


print(train_generator[0])


# In[5]:


VALIDATION_DIR = "E:/MaskImages/Test"

# Dataset de teste
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE,IMG_SIZE),
    class_mode='categorical',
    batch_size=32
)


# In[6]:


# Parametros do modelo
model = tf.keras.models.Sequential([
    # Primeira camada de convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Segunda camada de convolução
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Terceira camada de convolução
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Quarta camada de convolução
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Camada Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # Camada Dense
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# In[7]:


# Compilação do modelo
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)


# In[8]:


# Salvar o modelo
model.save("FaceMask_ModelV6.h5")


# In[9]:


# Plot doos graficos com metricas

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy [%]")
plt.legend(loc=0)
plt.figure()
plt.show()


# In[10]:


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc=0)
plt.figure()
plt.show()


# In[11]:


# Carregar o modelo
model = load_model("FaceMask_ModelV2.h5")


# In[12]:


# Teste do modelo com imagens individuais

path ='foto.jpg'
img=image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])
  
classification = model.predict(images, batch_size=10)
  
print(classification[0])


# In[ ]:



# Codigo para vizualizar como a rede neural se comporta em cada camada
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

img_path = 'foto.jpg'
img = load_img(img_path, target_size=(150, 150))  # Carregar imagem
x = img_to_array(img)  # Vetor com formato (150, 150, 3)
x = x.reshape((1,) + x.shape)  # reshape (1, 150, 150, 3)

# rescalar 1/255
x /= 255

successive_feature_maps = visualization_model.predict(x)

# Nomes de cada camada
layer_names = [layer.name for layer in model.layers[1:]]

# Print das imagens intermediarias
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
   if len(feature_map.shape) == 4:
       n_features = feature_map.shape[-1]  
       size = feature_map.shape[1]
       display_grid = np.zeros((size, size * n_features))
       for i in range(n_features):
           x = feature_map[0, :, :, i]
           x -= x.mean()
           x /= x.std()
           x *= 64
           x += 128
           x = np.clip(x, 0, 255).astype('uint8')
           display_grid[:, i * size : (i + 1) * size] = x
       scale = 20. / n_features
       plt.figure(figsize=(scale * n_features, scale))
       plt.title(layer_name)
       plt.grid(False)
       plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[ ]:




