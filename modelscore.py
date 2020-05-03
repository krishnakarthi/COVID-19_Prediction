import os
import json
import time
import numpy as np
import pandas as pd
import os
from glob import glob
import os.path as osp
import tensorflow as tf
import keras
import cv2
import base64
from keras.models import load_model

# Called when the deployed service starts
def init():
    global model
  
    
    print("init method has been invoked")
    

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'Covid_Final_Best_model.h5'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)
    # load models
    model = load_model(model_path)
    
    print("init method has been completed")

# Handle requests to the service
def run(data):
    try:
        print("run method has been invoked")
       
        imgInputData = imageDataPreProcess(data)
        prediction = predict(imgInputData)
        #Return prediction
        return prediction
    except Exception as e:
        error = str(e)
        print("ERROR :: ",error)
        return error

def imageDataPreProcess(inputData):
    #Convert inputdata to Json object
    jsonData=json.loads(inputData)
    #Get image bytes
    imageByte=jsonData["data"].encode("utf-8")
    #Get base64 bytes
    imgByteArray=base64.b64decode(imageByte)
    #Convert to byte Array
    imgByteArray = bytearray(imgByteArray)
    image = np.asarray(imgByteArray, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #Resize image to 150X150
    resized_arr = cv2.resize(image, (150, 150))
    #Normalize the image to 0-1 from 0-255
    X=resized_arr/255
    #Expand image dimension to (1,150,150,3)
    X = np.expand_dims(X, axis=0)
    return X


# Predict sentiment using the model
def predict(imgData):
    start_at = time.time()
    # Prediction
    prediction=model.predict(imgData)
    print("Predicted Value :",prediction[0][0])
    predictedValue=prediction[0][0]
    #Threshold limit value is 0.5
    if(predictedValue > 0.5):
        label="COVID-19"
    else:
        label="NORMAL"

    print("Predicted Label :",label)
    print("Process status :Completed") 
    return {"label": label, "prediction": str(predictedValue),
                "elapsed_time": time.time()-start_at}