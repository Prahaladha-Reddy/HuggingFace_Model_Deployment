
import streamlit as st
import requests
from scripts.s3 import *

## creating a file path for image
## python -m streamlit run streamlit_app.py --server.enableXsrfProtection false

## completed


API_URL = "http://127.0.0.1:8000/api/v1/"
headers = {
  'Content-Type': 'application/json'
}

st.title("ML Model Serving Over REST API")

model = st.selectbox("Select Model",
                     ["Sentiment Classifier", 
                      "Disaster Classifier", "Pose Classifier"])

if model=="Sentiment Classifier":
    text = st.text_area("Enter Your Movie Review")
    user_id = st.text_input("Enter user id", "prahaladreddy80@gmail.com")

    data = {"text": [text], "user_id": user_id}
    model_api = "get_sentiment"

elif model=="Disaster Classifier":
    text = st.text_area("Enter Your Tweet")
    user_id = st.text_input("Enter user id", "prahaladreddy80@gmail.com")

    data = {"text": [text], "user_id": user_id}
    model_api = "disaster_classifier"

elif model=="Pose Classifier":
    select_file = st.radio("Select the image source", ["Local", "URL"])

    if select_file=="URL":
        url = st.text_input("Enter Your Image Url")

    else:
        image = st.file_uploader("Upload the image", type=["jpg", "jpeg", "png"])
        file_name = "Models/images/temp.jpg"

        if image is not None:
            os.makedirs('Models/images', exist_ok=True)
            file_name = os.path.join('Models', 'images', 'temp.jpg')

            # Save the uploaded file
            try:
                with open(file_name, "wb") as f:
                    f.write(image.read())
                st.success(f"File saved at {file_name}")
            except Exception as e:
                st.error(f"Failed to save file: {e}")
        else:
            st.error("Please upload a file before proceeding.")


        url = upload_image_to_s3(file_name)


    user_id = st.text_input("Enter user id", "prahaladreddy80@gmail.com")

    data = {"url": [url], "user_id": user_id}
    model_api = "pose_classifier"

if st.button("Predict"):
    with st.spinner("Predicting... Please wait!!!"):
        response = requests.post(API_URL+model_api, headers=headers,
                                 json=data)
        
        output = response.json()

    st.write(output)