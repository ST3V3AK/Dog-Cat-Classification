import numpy as np
import cv2

IMG_SIZE = 180

def preprocess_image(image_np):
    image = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32) / 255.0   # same as Keras rescaling
    image = np.expand_dims(image, axis=0)
    return image