import streamlit as st
import time
from PIL import Image
import boto3
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer,pipeline
import torch


s3=boto3.client('s3')


bucket_name='tinybertsentimentanalysis'
local_filepath='tinybert_Downloaded'

def download_fir(local_path,s3_prefix):
  paginator = s3.get_paginator('list_objects_v2')
  for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
      if 'Contents' in result:
          for key in result['Contents']:
              s3_key = key['Key']

              file_path = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))

              os.makedirs(os.path.dirname(file_path), exist_ok=True)
              
              s3.download_file(bucket_name, s3_key, file_path)



st.title('Machine learning model prediction')

button=st.button('Download the model')
if button:
  with st.spinner('Downloading..😊'):  
    download_fir(local_filepath,'ml-models/')



model = AutoModelForSequenceClassification.from_pretrained(local_filepath)
tokenizer = AutoTokenizer.from_pretrained(local_filepath)


device=torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))


classifier=pipeline('sentiment-analysis',model=local_filepath,device=device)

input_text=st.text_input('Leave the comment here')
button=st.button('Analyze')
output=''
if button:
  with st.spinner('Analyzing..🔎'):  
    output=classifier(input_text)

st.write(output)