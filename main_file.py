import pickle

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

model = None
st.write("""
         # Image Classification
         """
         )

file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def load_model(location):
    return tf.keras.models.load_model(location)


model = load_model('best_CNN_model.h5')


def resize_test_image(test_img):
    img_test = Image.open(test_img)

    img_test.save("img.jpg")

    img = cv2.imread("img.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_RGB, (32, 32))
    resized_img = resized_img / 255.
    return resized_img


# classes = model.predict(images, batch_size=10)
def predict_test_image(test_img):
    resized_img = resize_test_image(test_img)
    prediction = model.predict(np.array([resized_img]))

    return prediction


def sort_prediction_test_image(test_img):
    prediction = predict_test_image(test_img)

    index = np.arange(0, 100)

    for i in range(100):
        for j in range(100):
            if prediction[0][index[i]] > prediction[0][index[j]]:
                temp = index[i]
                index[i] = index[j]
                index[j] = temp

    return index


def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict


metaData = unpickle('meta')
category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])


def df_top5_prediction_test_image(test_img):
    sorted_index = sort_prediction_test_image(test_img)
    prediction = predict_test_image(test_img)

    subCategory_name = []
    prediction_score = []

    k = sorted_index[:6]

    for i in range(len(k)):
        subCategory_name.append(subCategory.iloc[k[i]][0])
        prediction_score.append(round(prediction[0][k[i]], 2))

    df = pd.DataFrame(list(zip(subCategory_name, prediction_score)), columns=['Label', 'Probability'])

    return df


if file is None:
    st.text("Please upload an image file")
else:
    st.image(file, use_column_width=True)
    predictions = df_top5_prediction_test_image(file)
    print(predictions)

    st.write(predictions)
