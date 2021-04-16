# Environment Variables.
import os
# GUI Packages.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
import time

# from classify import pipeline_test

# AWS Packages.
import boto3
from botocore.exceptions import ClientError


# AWS Variables.
accessKeyID = os.environ["AWS_ACCESS_KEY_ID"]
secretAccessKey = os.environ["AWS_SECRET_ACCESS_KEY"]
s3BucketName = "ndelcoro-shape-classification-aws-bucket"
inputImageFileName = "shape.png"
resultsDataFileName = "results.txt"



# Other variables
timeout_seconds = 5


# ## Pipeline to upload images and then parse results

def parseAndShowResults(resultsDataFileName):
    print(resultsDataFileName)
    numbers=[]
    with open(resultsDataFileName, "r") as results:
        # Extract prediction results.
        for line in results:
            try:
                numbers.append(float(line[-10:-1]))
            except:
                pass
        # Find the prediction value with the highest prediction value.
        index_max = max(range(len(numbers)), key=numbers.__getitem__)

        # Display predicted value, prediction probability, and image of the hand-writtent digit that was classified.
    img = mpimg.imread(inputImageFileName)
    plt.imshow(img)
    class_labels = ['circle', 'square', 'star', 'triangle']
    plt.title('Predicted: ' + class_labels[index_max] + ' with a certainty of ' + str(numbers[index_max]))

## AWS Image Upload callback function and button ##

# Upload digit.png to S3 to produce the results.txt using lambda.
def awsImageUpload(buttonObject):

    client = boto3.client(
        's3',
        aws_access_key_id=accessKeyID,
        aws_secret_access_key=secretAccessKey
    )

    # Upload digit.png to S3.
    try:
        client.upload_file(inputImageFileName, s3BucketName, inputImageFileName)
    except ClientError as e:
        print("ERROR UPLOADING TO S3")
        return

    # Waiting and checking to see if the results.txt has been produced and placed in S3 from Lambda.
    timeout_timer = timeout_seconds
    while timeout_timer > 0:
        display("Uploading...")
        time.sleep(4)
        timeout_timer -= 4
        try:
            client.download_file(s3BucketName,resultsDataFileName,resultsDataFileName)
            display("Done! Parsing output...")
            break  #found results.txt
        except ClientError as e:
            pass

    # Removing input digit.png and output results.txt from S3.
    client.delete_object(Bucket=s3BucketName,Key=inputImageFileName)
    client.delete_object(Bucket=s3BucketName,Key=resultsDataFileName)

    # Display Results
    if timeout_timer <= 0:
        display("Timed out")
    else:
        parseAndShowResults(resultsDataFileName)

## Image upload callback function and button ##
def selectimage2upload(imageData):
    # Due to the file structure, image file name needs to be
    # extracted to access the bytes data of the image.
    imageFileName = list(imageData["new"].keys())[0]

    # Image bytes data.
    imageBytesData = imageData["new"][imageFileName]["content"]

    # Writing image file to current directory with "inputImageFileName".
    with open(inputImageFileName, "wb") as imageFile:
        imageFile.write(imageBytesData)

    # Displaying uploaded image in GUI.
    display(widgets.Image(value=imageBytesData))

    # Showing AWS GUI Components after image is uploaded.
    display(awsProgressRefreshRateSlider)
    display(awsUploadButton)

def sendImageToLocalPipeline(a):
    labeled_frame = pipeline_test(inputImageFileName)
    plt.imshow(labeled_frame,cmap='gray')
    plt.show()


# ### Creating GUI Components ###

def createDashBoard():
    # Allows the buttons to be accessed globally: Necessary
    # since some callback functions are dependent on these
    # widgets.
    global awsUploadButton
    global awsProgressRefreshRateSlider

    # AWS Image Upload Button.
    awsUploadButton = widgets.Button(
        description='Upload to AWS',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Upload to AWS'
    )
    awsUploadButton.on_click(awsImageUpload)

#     awsUploadButton.on_click(sendImageToLocalPipeline)

    # AWS Progress Refresh Rate Selector.
    awsProgressRefreshRateSlider = widgets.IntSlider(
        value=4,
        min=0,
        max=12,
        step=4,
        description='Timeout (sec)',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    # Display GUI.
    fileuploader = widgets.FileUpload(
        accept='.png',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False  # True to accept multiple files upload else False
    )
    fileuploader.observe(selectimage2upload,names='value')

    display(fileuploader)


