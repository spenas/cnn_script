import pandas as pd
import numpy as np
from numpy import genfromtxt
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import shutil  
import PIL

images_dir = os.path.abspath('images')
test_dir = os.path.abspath('test')
my_data = pd.read_csv('input.csv')

IMG_HEIGHT = 306
IMG_WIDTH = 306

new_model = keras.models.load_model('model_06_02.h5')
images = []

for i in my_data['Id']:
    
    path = os.path.join(images_dir, f'{i}')
    test_path = os.path.join(test_dir,  f'{i}')
    #check if file exists
    if not os.path.exists(test_path):
        shutil.move(path, test_dir)

    image = load_img(test_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    images.append(image)

images = np.vstack(images)
features = new_model.predict(images)
features = np.around(features).tolist()

final_predictions = []
for i in features:
    if i == [1.0, 0.0, 0.0]:
        final_predictions.append(2)
        
    elif i == [0.0, 1.0, 0.0]:
        final_predictions.append(1)
        
    elif i == [0.0, 0.0, 1.0]:
        final_predictions.append(0)




my_data['Category'] = final_predictions
my_data.to_csv('output.csv', index=False)
