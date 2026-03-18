import streamlit as st
from PIL import Image

st.title("Image Uploader")

uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.success("Image uploaded successfully!")
    st.image(image, caption="Uploaded Image", use_column_width=True)
else:
    st.info("Please upload an image file.")
