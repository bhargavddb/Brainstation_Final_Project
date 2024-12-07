import streamlit as st
import base64

# Set the page configuration
st.set_page_config(page_title="Financial Platform", layout="centered")

# Function to read image and convert to Base64
def encode_image_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Use the correct path to your image
image_path = "Images/landing_page.jpg"

try:
    encoded_image = encode_image_to_base64(image_path)

    # Inject the CSS to use the Base64 image as a background
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.8); /* Slight transparency */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.error("Background image not found. Please ensure the image exists in the 'static' directory.")

# Content for the app
st.title("Welcome to the Financial Platform")
st.subheader("Log in to access financial services")
username = st.text_input("Username", placeholder="Enter your username")
password = st.text_input("Password", placeholder="Enter your password", type="password")

if st.button("Log In"):
    st.success("Welcome!")
