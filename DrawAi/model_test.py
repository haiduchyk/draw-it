import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os
from PIL import ImageFont, ImageDraw, Image


def keras_predict(model, image):
    processed = keras_process_image(image)
    processed.reshape(28, 28)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return pred_probab, pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


model = load_model('models/QuickDraw.h5')

image = cv2.imread("test/axe5.png", cv2.IMREAD_GRAYSCALE)
image = (255 - image) / 255


result = keras_predict(model, image)
print(result)
