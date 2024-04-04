import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
st.header('Image Classification Model')
model = load_model('C:\Shruti\Classify\Image_Classify.keras')
data_cat = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
img_height = 128
img_width = 128
image = st.text_input('Enter Image name','mild.jpg')


image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('The level of Alzheimer is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))