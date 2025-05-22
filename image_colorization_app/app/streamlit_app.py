# app/streamlit_app.py
import streamlit as st
from PIL import Image
from app.inference import load_model, colorize_image

st.title("🎨 Automatic Image Colorization")
st.write("Upload a grayscale image to see its colorized version using a deep learning model!")

uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.subheader("Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Grayscale Image", use_container_width=True)
        with st.spinner("Loading model..."):
            model = load_model()
        st.success("Model loaded!")

    with col2:
        with st.spinner("Colorizing image..."):
            colorized = colorize_image(image, model)
            colorized = colorized.resize(image.size)
        st.image(colorized, caption="Colorized Image", use_container_width=True)
        st.success("Image colorized!")
