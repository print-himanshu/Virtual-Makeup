import tensorflow as tf
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as tfback


from util import _get_available_gpus, facial_landmark, color_lip
from util import hair_mask, hair_colorimg, empty, competitionMetric2


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)


# GPU preprocessing----------------------------------------------------------------------------
tfback._get_available_gpus = _get_available_gpus
gpus = tf.compat.v1.config.experimental.list_physical_devices("GPU")
if gpus:
  # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*4))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# GPU preprocessing----------------------------------------------------------------------------


model = tf.keras.models.load_model('hair_weights/hair_unet.h5', 
                custom_objects={'competitionMetric2': competitionMetric2})


# Creating two seperate Window------------------------------------------------------------------------
cv.namedWindow('Hair Color')
cv.resizeWindow("Hair Color", 640 , 250)
cv.createTrackbar("Blue", "Hair Color", 0, 255 ,empty)
cv.createTrackbar("Green", 'Hair Color', 0 , 255, empty)
cv.createTrackbar("Red", "Hair Color", 0 , 255, empty)


cv.namedWindow('Lip Color')
cv.resizeWindow("Lip Color", 640 , 250)
cv.createTrackbar("Blue", "Lip Color", 0, 255 ,empty)
cv.createTrackbar("Green", 'Lip Color', 0 , 255, empty)
cv.createTrackbar("Red", "Lip Color", 0 , 255, empty)
#Creating two seperate window -------------------------------------------------------------------------



lip_image_path = os.path.join(os.getcwd(), '00000007.jpg')
lip_image = cv.imread(lip_image_path)
landmark_point = np.array(facial_landmark(lip_image))


hair_image_path = os.path.join(os.getcwd(), 'Frame00267.jpg')
hair_image = cv.imread(hair_image_path)
hair_image = cv.resize(hair_image, (256,256))

mask = hair_mask(hair_image, model)

while True:
    lip_b = cv.getTrackbarPos("Blue", "Lip Color")
    lip_g = cv.getTrackbarPos("Green", "Lip Color")
    lip_r = cv.getTrackbarPos("Red", "Lip Color")


    image_lip_colored = color_lip(lip_image,landmark_point,  lip_b,lip_g,lip_r)
    cv.imshow('Lip Color',image_lip_colored)
    
    hair_b = cv.getTrackbarPos("Blue", "Hair Color")
    hair_g = cv.getTrackbarPos("Green", "Hair Color")
    hair_r = cv.getTrackbarPos("Red", "Hair Color")


    hair_colored = hair_colorimg(hair_image,mask,  hair_b,hair_g,hair_r)
    hair_colored = cv.resize(hair_colored, (0,0), None , 2 ,2 )
    cv.imshow('Hair Color',hair_colored)
    cv.waitKey(0)






    