import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm

def extract_features(image_folder):

    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    features = {}

    for img_name in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        image_id = img_name.split(".")[0]
        features[image_id] = feature

    return features