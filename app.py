from scripts.data_model import *
from scripts.s3 import *
from fastapi import FastAPI
from fastapi import Request
import uvicorn
import os
import torch
import torchvision
from transformers import pipeline
import warnings
import time
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Download ml models
bucket_name='tinybertsentimentanalysis'
local_dir='Models'
model_name='tinybert_sentiment_analysis'

if not os.path.isdir(local_dir):
  download_s3_bucket(bucket_name, local_dir)

## download complete

## Model creation
from transformers import AutoImageProcessor
model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

sentiment_model=pipeline('text-classification',model="Models/tinybert_Downloaded",device=device)

twitter_model=pipeline('text-classification',model="Models/disaster_management_classification",device=device)

pose_model=pipeline('image-classification',model="Models/pose_classification",device=device,image_processor=image_processor)


## Model creation complete
app=FastAPI()

@app.get('/')
def read_root():
  return "hello"

@app.post('/api/v1/get_sentiment')
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert_Downloaded",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)

    return output



@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = twitter_model(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="disaster_management_classification",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)

    return output

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput):
    start = time.time()
    # print(data)
    urls = [str(x) for x in data.url]
    output = pose_model(urls)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]

    output = ImageDataOutput(model_name="pose_classification",
                           url = data.url,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)
    
    return output











if __name__=='__main__':
  uvicorn.run(app=app,port=8000,reload=True)