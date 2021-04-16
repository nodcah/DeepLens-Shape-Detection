#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import json
import time
import numpy as np
import cv2

import traceback

import sys
import onnxruntime as rt


def preprocess(image):
    # TODO only pull out green pixels from the image
    h, s, v, h1, s1, v1 = 42, 60, 0, 80, 255, 255
    input_shape = image.shape

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h, s, v]), np.array([h1, s1, v1])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    kernel = np.ones((3, 3), np.uint)
    gray = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    gray = cv2.bitwise_not(gray)
    # TODO: Crop the center square of the given image/frame.
    # Image/frame dimension is around 16:9 ratio, similar to a movie screen.
    # Crop the image/frame to get the center square of it.

    h, w = gray.shape
    crop_img = gray[0:int(h), int((w - h) / 2):int((w + h) / 2)]

    # Resizing to make it compatible with the model input size.
    resized_img = cv2.resize(crop_img, (64, 64)).astype(np.float32) / 255
    img = np.reshape(resized_img, (1, 64, 64, 1))
    return img


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def makeInferences(sess, input_img):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onx = sess.run([output_name], {input_name: input_img})[0]
    scores = softmax(pred_onx)
    return scores


# ====================== LOCAL VERSION ======================


def pipeline_test(image_file_name):
    """ Run the DeepLens inference loop frame by frame"""
    try:
        model_directory = "/Users/nodcah/Downloads"
        model_name = "model.onnx"  # onnx-model

        # When the ONNX model is imported via DeepLens console, the model is copied
        # to the AWS DeepLens device, which is located in the "/opt/awscam/artifacts/".
        model_file_path = os.path.join(model_directory, model_name)
        sess = rt.InferenceSession(model_file_path)

        cap = cv2.VideoCapture(0)

        while True:
            # Get a frame 
            #           frame = cv2.imread(image_file_name)
            frame = cap.read()[1]
            # Preprocess the frame to crop it into a square and
            # resize it to make it the same size as the model's input size.
            input_img = preprocess(frame)

            # Inference.
            # print(input_img.shape)
            inferences = makeInferences(sess, input_img)
            inference = np.argmax(inferences)

            class_labels = ['●', '■', '★', '▲']
            # class_labels = ['circle', 'square', 'star', 'triangle']
            # print(f'Labels={class_labels}')
            print(f'inferences:')
            for label in range(len(class_labels)):
                print(f'  {class_labels[label]}: {inferences[0][label]}')
            print(f'Inference = {class_labels[inference]}')
            print()

            cv2.imshow('frame', input_img[0, :, :, :])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(.5)

        cap.release()
        cv2.destroyAllWindows()
        return input_img[0, :, :, :]


    except Exception:  # as ex:
        # Outputting error logs as "MQTT messages" to AWS IoT.
        # print('Error in lambda {}'.format(ex))
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("error details:" + str(exc_type) + str(fname) + str(exc_tb.tb_lineno))
        print(traceback.format_exc())


inputImageFileName = "shape.png"
pipeline_test(inputImageFileName)
# ### Creating GUI Components ###
# 
# 

# In[8]:
