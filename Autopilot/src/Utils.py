from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pickle


def load_from_pickle(pickle_features_path, pickle_labels_path):
    with open(pickle_features_path, "rb") as f:
        features = np.array(pickle.load(f))
    with open(pickle_labels_path, "rb") as f:
        labels = np.array(pickle.load(f))
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    features, labels = shuffle(features, labels)
    print("Pickles loaded successfully..")
    return features, labels


def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    return resized


def print_model_summary(model_path, test_x, test_y):
    model = load_model(model_path)
    model.compile(optimizer="Adam", loss='mse')
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss for model (' + model_path + ') is: ' + str(score))
    print('Test accuracy for model (' + model_path + ') is: ' + str(1 - score))


def keras_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle = float(model.predict(processed, batch_size=1))
    steering_angle = steering_angle * 100
    return steering_angle
