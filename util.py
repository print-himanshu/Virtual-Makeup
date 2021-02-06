import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
import keras.backend as K

import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
import os


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Input, concatenate
from tensorflow.keras.models import Model

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

def U_NetModel(input_shape):

    X_input = Input(input_shape)

    ######## Down-Sampling Part
    d1 = Conv2D(filters = 16, kernel_size = (3,3), 
                name = "downsamping-1a", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(X_input)
    d1 = Dropout(0.1)(d1)
    d1 = Conv2D(filters = 16, kernel_size = (3,3), 
                name = "downsampling-1b", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d1)
    p1 = MaxPooling2D(pool_size= (2,2))(d1)




    d2 = Conv2D(filters = 32, kernel_size = (3,3), 
                name = "downsamping-2a", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(p1)
    d2 = Dropout(0.1)(d2)
    d2 = Conv2D(filters = 32, kernel_size = (3,3), 
                name = "downsampling-2b", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d2)
    p2 = MaxPooling2D(pool_size= (2,2))(d2)




    d3 = Conv2D(filters = 64, kernel_size = (3,3), 
                name = "downsamping-3a", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(p2)
    d3 = Dropout(0.1)(d3)
    d3 = Conv2D(filters = 64, kernel_size = (3,3), 
                name = "downsampling-3b", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d3)
    p3 = MaxPooling2D(pool_size= (2,2))(d3)





    d4 = Conv2D(filters = 128, kernel_size = (3,3), 
                name = "downsamping-4a", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(p3)
    d4 = Dropout(0.1)(d4)
    d4 = Conv2D(filters = 128, kernel_size = (3,3), 
                name = "downsampling-4b", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d4)
    p4 = MaxPooling2D(pool_size= (2,2))(d4)






    d5 = Conv2D(filters = 256, kernel_size = (3,3), 
                name = "downsamping-5a", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(p4)
    d5 = Dropout(0.1)(d5)
    d5 = Conv2D(filters = 256, kernel_size = (3,3), 
                name = "downsampling-5b", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d5)
    d5 = Dropout(0.1)(d5)
    d5 = Conv2D(filters = 256, kernel_size = (3,3), 
                name = "downsampling-5c", padding = "same", 
                kernel_initializer= "he_normal", activation = "relu")(d5)
    


    u6 = Conv2DTranspose(filters = 128, kernel_size = (2,2), 
                         strides = (2,2), padding = "same",
                         name = "upsampling-6a")(d5)
    u6 = concatenate([u6, d4])
    c6 = Conv2D(filters = 128, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-6b")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(filters = 128, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-6c")(c6)

    
    
    
    u7 = Conv2DTranspose(filters = 64, kernel_size = (2,2), 
                         strides = (2,2), padding = "same",
                         name = "upsampling-7a")(c6)
    u7 = concatenate([u7, d3])
    c7 = Conv2D(filters = 64, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-7b")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(filters = 64, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-7c")(c7)

    

    
    
    u8 = Conv2DTranspose(filters = 32, kernel_size = (2,2), 
                         strides = (2,2), padding = "same",
                         name = "upsampling-8a")(c7)
    u8 = concatenate([u8, d2])
    c8 = Conv2D(filters = 32, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-8b")(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(filters = 32, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-8c")(c8)

    

    
    
    u9 = Conv2DTranspose(filters = 16, kernel_size = (2,2), 
                         strides = (2,2), padding = "same",
                         name = "upsampling-9a")(c8)
    u9 = concatenate([u9, d1])
    c9 = Conv2D(filters = 16, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-9b")(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(filters = 16, kernel_size = (2,2),
                activation = "relu", padding = 'same',
                kernel_initializer = 'he_normal', name = "upsampling-9c")(c9)



    output = Conv2D(filters = 1, kernel_size = (1,1), name = "output", activation = "sigmoid")(c9)

    model = Model(inputs = [X_input], outputs = [output], name = "U_NET_MODEL")
    
    return model


def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def competitionMetric2(true, pred): #any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

def facial_landmark(image):
    detector = dlib.get_frontal_face_detector()
    imgGray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    faces = detector(imgGray)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    landmark_point = []
    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()

        #img_original__ = cv.rectangle(img_original, (x1,y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_point.append((x,y))
            #cv.circle(img_original__, (x,y), 5, (50, 50 , 255), cv.FILLED)
            #cv.putText(img_original__, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

    return landmark_point

def create_box(img, points, scale = 5, masked = False, cropped = True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv.fillPoly(mask, [points], (255,255,255))
        img = cv.bitwise_and(img, mask)

    if cropped:
        bbox = cv.boundingRect(points)
        x,y,w,h = bbox

        img_crop = img[y:y+h, x:x+w]
        img_crop = cv.resize(img_crop, (0,0), None , scale ,scale)
        return img_crop
        
    return mask

def color_lip(original_image,landmark_point,  b,g,r):
    lip_mask = create_box(original_image, landmark_point[48:61], masked = True, cropped = False)
    lip_color = np.zeros_like(lip_mask)
    lip_color[:] = b, g , r

    colored_lip = cv.bitwise_and(lip_mask, lip_color)
    colored_lip = cv.GaussianBlur(colored_lip, (7,7), 10)
    img_color_lip = cv.addWeighted(original_image, 1, colored_lip, 0.4, 0)
    return img_color_lip

def empty(a):
    pass

def hair_mask(img_train, model):
    try:
        os.mkdir('dataset/prediction')
    except OSError as e:
        pass

    
    plt.imsave('dataset/prediction/img-1.jpg', img_train)

    img_train = img_train / 255.

    img_train = np.expand_dims(img_train, 0)
    predict = model.predict(img_train, verbose = 1)
    prt = (predict > 0.4).astype(np.uint8)
    plt.imsave('dataset/prediction/mask-1.jpg', np.squeeze(prt), cmap = 'gray')


    mask = cv.imread('dataset/prediction/mask-1.jpg',0)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    return mask

def hair_colorimg(input_img,mask, b , g ,r):
    hair_color = np.zeros_like(mask)
    hair_color[:] = b ,g ,r


    color_hair = cv.bitwise_and(mask, hair_color)
    color_hair = cv.GaussianBlur(color_hair, (7,7), 10)
    colored_hair = cv.addWeighted(input_img, 1, color_hair, 0.4, 0)

    return colored_hair