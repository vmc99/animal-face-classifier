import streamlit as st
from PIL import Image
import requests
from model import transform_image, predict_image
import gc

#Enable garbage collection
gc.enable()

st.title('Animal Face Classifier ')
st.write("This is a simple image classification web app to predict animal faces into three classes `cats, dogs and wild`.")

image = None

# Demo 
st.write('### Select an image for Demo')
menu = ['Select an image', 'Image 1', 'Image 2']
choice = st.selectbox('Select', menu)


# File upload method
st.write('### Upload your image')
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

st.write("## Or")

# URL method
st.write('### Enter the image URL')
url =  st.text_input('URL')


if file:
    image = Image.open(file)
    st.write('### `image`')
    st.image(image, width=400)


elif url:
    # image = requests.get(path).content
    # img = Image.open(image)
    image = Image.open(requests.get(url, stream=True).raw)
    if image.format == 'JPEG' or 'PNG' or 'JPG' :
        st.write('### `image`')
        st.image(image, width=400)
    else:
        st.error('Image format not supported')

elif choice == 'Image 1':
    image = Image.open('demo_images/flickr_cat_000018.jpg')
    st.write('### `image`')
    st.write('image format :',image.format)
    st.image(image, width=400)

elif choice == 'Image 2':
    image = Image.open('demo_images/flickr_dog_000231.jpg')
    st.write('### `image`')
    st.write('image format :',image.format)
    st.image(image, width=400)




clicked = st.button('Run')

if clicked:
    if image is None:
        st.error('Please upload an image file or Enter the image url')
    else:
        if len(image.getbands()) == 3:
            with st.spinner(text='In progress'):
                image_transformed = transform_image(image)
                class_label , confidence = predict_image(image_transformed)

            st.info('Run Successful !')
            if confidence > 50:
                st.write('### `Prediction` : ', class_label)
                st.write('### `Confidence` : {}%'.format(confidence)) 
                del image, class_label, confidence
            else:
                st.write('### `Prediction` : Unable to predict')
                st.write('### `Confidence` : {}%'.format(confidence)) 
                del image, confidence

        else:
            st.write('### `Prediction` : image not compatible')
            st.write('### `Confidence` : nan')
            del image 



gc.collect()
