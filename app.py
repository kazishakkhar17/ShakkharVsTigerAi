import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from urllib.request import urlopen
import tensorflow as tf

# Load your trained model (make sure the model file is in the same folder)
model = tf.keras.models.load_model('tiger_shakkhar_model.keras')

classes = ['Shakkhar', 'Tiger']

# Public image URLs (replace with your actual image URLs)
TIGER_IMAGE_URL = "https://t4.ftcdn.net/jpg/05/71/21/43/360_F_571214391_jnT6Rsg2M7VtsOs0MF4i1VFKAqWlHI47.jpg"
SHAKKHAR_IMAGE_URL = "https://scontent.frjh4-1.fna.fbcdn.net/v/t39.30808-1/471496368_1803695170458604_1398081511664333487_n.jpg?stp=dst-jpg_s200x200_tt6&_nc_cat=100&ccb=1-7&_nc_sid=e99d92&_nc_eui2=AeFYcYQWsEwneG0RwlOrpsoSIbCslynZu_AhsKyXKdm78BJ_WnpX-4plrgzrSSbSKmVpYrYGIe16qDkBAdFI63XD&_nc_ohc=9UnEL6DLIy4Q7kNvwETz2GP&_nc_oc=Adn-FnAfbFNHOANE5-eqA31RzxyRbds2YCUe10yeXVYoTdvNjlTcX5yGHL28O76uc2E&_nc_zt=24&_nc_ht=scontent.frjh4-1.fna&_nc_gid=F2IcUAAP9gFdqLyjAoS1bQ&oh=00_AfTVtDBLIn0mOzGMVoIOutyTb6BC6h46WRSRHF6a-gvP3Q&oe=687C437E"

# --- Custom CSS for styling ---
st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
        padding: 2rem;
        border-radius: 15px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0 8px 24px rgba(149, 157, 165, 0.2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .image-container {
        display: flex;
        gap: 2rem;
        justify-content: center;
        margin-top: 1rem;
    }
    .square-img {
        width: 250px;
        height: 250px;
        object-fit: cover;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        border: 3px solid #3b82f6; /* Blue border */
    }
    .title {
        text-align: center;
        color: #1e293b;
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
    }
    .caption {
        text-align: center;
        font-style: italic;
        color: #475569;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">Handwriting Classification: Tiger vs Shakkhar</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Show uploaded image preview
    uploaded_img = Image.open(uploaded_file).convert("RGB")  # <== FIXED here
    uploaded_img_square = uploaded_img.resize((250, 250))

    # Convert uploaded image for model prediction
    img_for_pred = uploaded_img.resize((224, 224))
    img_array = image.img_to_array(img_for_pred) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = classes[1] if prediction >= 0.5 else classes[0]
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.write(f"### Prediction: **{label}**")
    st.write(f"### Confidence: **{confidence:.2f}**")

    # Load related image from URL and resize for square box
    related_img_url = TIGER_IMAGE_URL if label.startswith('Tiger') else SHAKKHAR_IMAGE_URL
    related_img = Image.open(urlopen(related_img_url)).resize((250, 250))

    # Display both images side by side with styling
    st.markdown('<div class="image-container">', unsafe_allow_html=True)

    st.image(uploaded_img_square, caption="Uploaded Image", width=250)
    st.image(related_img, caption=label, width=250)

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload an image to classify.")
