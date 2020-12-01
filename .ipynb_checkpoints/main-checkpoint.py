import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import requests
from PIL import Image
from PIL import ImageDraw, ImageFont
import cv2
import io

SUBSCRIPTIONKEY='f37650dbe4bd4d35966e156fc524ab84'
assert SUBSCRIPTIONKEY
face_api_url ='https://20201201motiduki.cognitiveservices.azure.com/face/v1.0/detect'

st.title('顔認識アプリ')

uploaded_file=st.file_uploader("Choose an image...",type='jpg')
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    

    with io.BytesIO() as output:
        img.save(output,format="JPEG")
        binary_img=output.getvalue()
    headers = {'Ocp-Apim-Subscription-Key':  SUBSCRIPTIONKEY,
                'Content-Type':'application/octet-stream'}
    params = {
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'returnFaceId': 'true'
    }

    res = requests.post(face_api_url, params=params,
                             headers=headers,data=binary_img )
    results=res.json()

    for result in results:
        rect=result['faceRectangle']
        age=result[ 'faceAttributes']['age']
        gender=result[ 'faceAttributes']['gender']
        smile=result[ 'faceAttributes']['smile']
        age='age:'+str(age)
        gender='gender:'+str(gender)
        smile='smile:'+str(smile)
        draw=ImageDraw.Draw(img)
        draw.rectangle([(rect['left'],rect['top']),(rect['left']+rect['width'],rect['top']+rect['height'])],fill=None,outline='green',width=5)
        font = ImageFont.truetype("arial.ttf", 50)
        draw.text((rect['left'],rect['top']),age, 250,font=font)
        draw.text((rect['left'],rect['top']+40), gender, 250,font=font)
        draw.text((rect['left'],rect['top']-40), smile, 250,font=font)

    img
    st.image(img,caption='Uploaded Image.',use_column_width=True)
df = pd.read_excel('nenkin.xlsx',skiprows=10,skipfooter=24,sheet_name=21,index_col='Unnamed: 0')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
@st.cache

    
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    
    return data
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

df = pd.read_excel('nenkin.xlsx',skiprows=10,skipfooter=24,sheet_name='静岡',index_col='Unnamed: 0')
df = df.rename(columns={'Unnamed: 1':'月額', 'Unnamed: 3':'-','Unnamed: 5':'1号全額','Unnamed: 6':'1号半額','Unnamed: 7':'2号全額','Unnamed: 8':'2号半額','Unnamed: 9':'船員等全額','Unnamed: 10':'船員等半額'})
st.dataframe(df)
st.subheader('Raw data')
st.write(data)
st.subheader('Number of pickups by hour')
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)
hour_to_filter = 17
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)
hour_to_filter = st.slider('hour', 0, 23, 17)

