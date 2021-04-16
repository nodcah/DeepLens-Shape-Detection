# import os
# import json
# import time
# import numpy as np
# import cv2
# import traceback

# import sys
# import onnxruntime as rt

# lambda
import sys

sys.path.append("./packages")
import cv2
import os
import numpy as np
from datetime import datetime
import boto3
import onnxruntime as rt

PAD = 75


def preprocess(image):
    # only pull out green pixels from the image
    h, s, v, h1, s1, v1 = 40, 45, 0, 80, 255, 255
    input_shape = image.shape

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h, s, v]), np.array([h1, s1, v1])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    kernel = np.ones((3, 3), np.uint)
    gray = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        c = max(contours, key=lambda con: cv2.contourArea(con))
        if cv2.contourArea(c) < 3000:
            raise ValueError
        # Find bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        box = gray[y - PAD:y + h + PAD, x - PAD:x + w + PAD]
        if len(box) < 10:
            raise ValueError
        gray = cv2.bitwise_not(box)
    except ValueError:
        gray = cv2.bitwise_not(gray)

    # Crop the center square of the given image/frame.
    # Image/frame dimension is around 16:9 ratio, similar to a movie screen.
    # Crop the image/frame to get the center square of it.
    h, w = gray.shape
    if w >= h:
        crop_img = gray[0:int(h), int((w - h) / 2):int((w + h) / 2)]
    else:
        crop_img = gray[int((h - w) / 2):int((w + h) / 2), 0:int(w)]

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

'''
def pipeline_test(image_file_name):
    model_directory = "/Users/nodcah/Downloads"
    model_name = "model.onnx"  # onnx-model


    # When the ONNX model is imported via DeepLens console, the model is copied
    # to the AWS DeepLens device, which is located in the "/opt/awscam/artifacts/".
    model_file_path = os.path.join(model_directory, model_name)
    sess = rt.InferenceSession(model_file_path)

    # Get a frame 
    frame = cv2.imread(image_file_name)
    #print(frame.shape)
    # Preprocess the frame to crop it into a square and
    # resize it to make it the same size as the model's input size.
    input_img = preprocess(frame)

    # Inference.
    inferences = makeInferences(sess, input_img)
    inference = np.argmax(inferences)
    
    class_labels = ['●', '■', '★', '▲']
    print(f'inferences:')
    for label in range(len(class_labels)):
        print(f'  {class_labels[label]}: {inferences[0][label]}')
    print(f'Inference = {class_labels[inference]}')
    
    return input_img[0,:,:,:]
#pipeline_test('shape.png')
'''


# ====================== Lambda Version ======================

def lambda_handler(event, context):
    s3_bucket_name = "ndelcoro-shape-classification-aws-bucket"
    lambda_tmp_directory = "/tmp"
    #     lambda_tmp_directory = "."
    model_file_name = "model.onnx"
    input_file_name = "shape.png"
    output_file_name = "results.txt"

    # Making probability print-out look pretty.
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    try:
        # Download test image and model from S3.
        client = boto3.client('s3')
        client.download_file(s3_bucket_name, input_file_name, os.path.join(lambda_tmp_directory, input_file_name))
        client.download_file(s3_bucket_name, model_file_name, os.path.join(lambda_tmp_directory, model_file_name))
    except:
        pass

    # Import input image in grayscale and preprocess it.
    #     image = Image.open(os.path.join(lambda_tmp_directory, input_file_name))
    #     img = np.array(image)[:,:,0:3]

    frame = cv2.imread(os.path.join(lambda_tmp_directory, input_file_name))
    processed_image = preprocess(frame)

    # Make inference using the ONNX model.
    sess = rt.InferenceSession(os.path.join(lambda_tmp_directory, model_file_name))
    inferences = makeInferences(sess, processed_image)

    # Output probabilities in an output file.
    f = open(os.path.join(lambda_tmp_directory, output_file_name), "w+")
    f.write("Predicted: %d \n" % np.argmax(inferences))
    for i in range(4):
        f.write("class=%s ; probability=%f \n" % (i, inferences[0][i]))
    f.close()

    # Get today's date and append to the filename.
    current_date_time = str(datetime.now())

    try:
        # Upload the output file to the S3 bucket.
        client.upload_file(os.path.join(lambda_tmp_directory, output_file_name), s3_bucket_name, output_file_name)
    except:
        pass


# lambda_handler(None,None)


# ====================== DeepLens Version ======================
'''
import mo
import greengrasssdk
from utils import LocalDisplay

import awscam

def lambda_handler(event, context):
    """Empty entry point to the Lambda function invoked from the edge."""
    return

# Create an IoT client for sending to messages to the cloud.
# client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ["AWS_IOT_THING_NAME"])


def infinite_infer_run():
    """ Run the DeepLens inference loop frame by frame"""
    try:
        model_directory = "/opt/awscam/artifacts/"
        model_name = "model"  # onnx-model

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # When the ONNX model is imported via DeepLens console, the model is copied
        # to the AWS DeepLens device, which is located in the "/opt/awscam/artifacts/".
        model_file_path = os.path.join(model_directory, model_name)
        sess = rt.InferenceSession(model_file_path)

        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')

            # Preprocess the frame to crop it into a square and
            # resize it to make it the same size as the model's input size.
            input_img = preprocess(frame)

            # Inference.
            inferences = makeInferences(sess, input_img)
            inference = np.argmax(inferences)

            # TODO: Add the label of predicted digit to the frame used by local display.
            # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
            # for more information about the cv2.putText method.
            # Method signature: image, text, origin, font face, font scale, color, and thickness 
            frame = cv2.putText(frame, str(inference), (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 15)

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)

            # Outputting the result logs as "MQTT messages" to AWS IoT.
            cloud_output = {}
            cloud_output["scores"] = inferences.tolist()
            print(inference, cloud_output)

    except Exception:  # as ex:
        # Outputting error logs as "MQTT messages" to AWS IoT.
        # print('Error in lambda {}'.format(ex))
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("error details:" + str(exc_type) + str(fname) + str(exc_tb.tb_lineno))
        print(traceback.format_exc())


infinite_infer_run()

'''
