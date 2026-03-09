import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import pygame
import uuid
import os
import time

#GENERATED CAPTION AND CAPTION LABEL 
st.markdown("""
<style>
/* Center ALL content */
.block-container {
    max-width: 1100px;
    margin: auto;
    text-align: center;
}

/* Center columns content */
[data-testid="column"] {
    text-align: center;
}


/* Center file uploader */
[data-testid="stFileUploader"] {
    text-align: center;
}

/* Center images */
img {
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
}

/* Center markdown text */
div[data-testid="stMarkdownContainer"] {
    text-align: center;
}
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600&family=Source+Sans+3:wght@400&display=swap');

/* Label — Generated Caption*/
.caption-label{
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 28px;
    color: white;
}

/* Caption Text */
.caption-text{
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 300;
    font-size: 20px;
    color: white;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)
#PREVIEW AND UPLOAD IMAGE 
st.markdown("""
<style>
/* Center only buttons */
.stButton > button {
    display: block;
    margin: 0 auto;
}
/* Center ALL content */
.block-container {
    max-width: 1100px;
    margin: auto;
    text-align: center;
}

/* Center columns content */
[data-testid="column"] {
    text-align: center;
}


/* Center file uploader */
[data-testid="stFileUploader"] {
    text-align: center;
}

/* Center images */
img {
    margin-left: auto !important;
    margin-right: auto !important;
    display: block !important;
}

/* Center markdown text */
div[data-testid="stMarkdownContainer"] {
    text-align: center;
}
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

.section-header{
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 26px;
    color: white;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


#HEADING AND SUBHEADING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Lato:ital@1&display=swap');

/* Force Poppins everywhere */
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif !important;
}

/* MAIN HEADING */
.title{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 400 !important;
    font-size: 64px !important;
    text-align: center !important;
    color: white !important;
    letter-spacing: 1px;
}

/* SUBHEADING */
.subtitle{
    font-family: 'Lato', sans-serif !important;
    font-style: italic !important;
    font-size: 22px;
    text-align: center;
    color: rgba(255,255,255,0.9);
    margin-top: -10px;
}

/* Background */
body{
    background: linear-gradient(135deg,#667eea,#764ba2);
}
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="title">Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let Artificial Intelligence describe it automatically</div>', unsafe_allow_html=True)

st.markdown("""
<hr style="
    border: none;
    height: 2px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.6), transparent);
    margin: 30px 0 20px 0;
">
""", unsafe_allow_html=True)


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "caption" not in st.session_state:
    st.session_state.caption = ""

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# -----------------------------
# Animated Gradient Background
# -----------------------------
st.markdown("""
<style>
@keyframes gradientFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(-45deg, #667eea, #764ba2, #6dd5ed, #2193b0);
    background-size: 400% 400%;
    animation: gradientFlow 15s ease infinite;
}

.main-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 0.2em;
}

.subtitle {
    text-align: center;
    color: #eef2ff;
    margin-bottom: 1.5em;
    font-size: 18px;
}

.caption-text {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.6;
    margin-top: 8px;
    margin-bottom: 18px;
    letter-spacing: 0.3px;
}

.footer {
    text-align: center;
    color: #e0e7ff;
    font-size: 14px;
    margin-top: 3rem;
}
.section-header {
    background: rgba(255,255,255,0.15);
    padding: 16px 20px;
    border-radius: 18px;
    color: white;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Caption Model (cached)
# -----------------------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()

# -----------------------------
# Voice Function (Natural Female)
# -----------------------------
def speak_text(text):
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.music.unload()
    os.remove(filename)

# -----------------------------
# Caption Generator
# -----------------------------
def generate_caption(image: Image.Image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip().capitalize()


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-header"> Upload Image</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

with right:
    st.markdown(
        '<div class="section-header"> Preview</div>',
        unsafe_allow_html=True
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.last_image = image
        st.image(image, width="stretch")
    elif st.session_state.last_image is not None:
        st.image(st.session_state.last_image, width="stretch")
    else:
        st.info("Image preview will appear here")

# -----------------------------
# Generate Button
# -----------------------------
st.write("")
center = st.columns([1,2,1])[1]
with center:
    generate_btn = st.button("✨ Generate Caption", use_container_width=True)

# -----------------------------
# Generate Caption
# -----------------------------
if generate_btn and uploaded_file:
    with st.spinner("Analyzing image with AI..."):
        st.session_state.caption = generate_caption(st.session_state.last_image)

elif generate_btn and not uploaded_file:
    st.warning("⚠️ Please upload an image first")

# -----------------------------
# Caption Output (Perfect Layout)
# -----------------------------
st.markdown(
    '<div class="caption-label"> Generated Caption-</div>',
    unsafe_allow_html=True
)

if st.session_state.caption:
    st.markdown(
        f'<div class="caption-text">{st.session_state.caption}</div>',
        unsafe_allow_html=True
    )

    # Centered button below caption
    btn_col = st.columns([1,2,1])[1]
    with btn_col:
        if st.button("🔊 Read Caption"):
            speak_text(st.session_state.caption)
else:
    st.markdown(
        '<div class="caption-text">No caption generated yet</div>',
        unsafe_allow_html=True
    )

st.markdown("""
<hr style="
    border: none;
    height: 2px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.6), transparent);
    margin: 30px 0 20px 0;
">
""", unsafe_allow_html=True)

# -----------------------------
# Info / About Project Section
# -----------------------------
st.write("")

st.markdown("""

<div style="
    background: rgba(255,255,255,0.12);
    padding: 18px 22px;
    border-radius: 12px;
    color: white;
    font-size: 16px;
    line-height: 1.7;
">

<b> About</b><br>

This system leverages a pretrained <b>Vision–Language Transformer</b> model to automatically understand images and generate natural language captions.
It integrates <b>Computer Vision</b> and <b>Natural Language Processing</b> to convert visual information into human-readable text.
<br><br>

<b>Features</b><br>
• Real-time caption generation<br>
• Interactive web interface<br>
• Voice-enabled caption reading<br><br>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<style>
.footer a{
    color: #ffffff !important;
    text-decoration: underline !important;
    font-weight: 600;
}
.footer a:hover{
    color: #a8e6ff !important;
    text-shadow: 0 0 4px rgba(205,180,255,0.9);
}
</style>

<div class="footer" style="
    text-align:center;
    color: rgba(255,255,255,0.85);
    font-size:16px;
    line-height:2.2;
    letter-spacing:0.4px;
    margin-top:50px;
    font-family: 'Source Sans 3', sans-serif;
">

AI Vision–Language Captioning System<br>

Developed by 
<a href="https://www.linkedin.com/in/vedant-singh-rathore-10aa11377" target="_blank">
Vedant Singh Rathore
</a> 
& 
<a href="https://www.linkedin.com/in/vansh-dwivedi-45a634349" target="_blank">
Vansh Dwivedi
</a>

Contact: 
<a href="mailto:vedantsinghrathore@gmail.com">
ivedantsinghrathore@gmail.com
</a>
&nbsp;|&nbsp;
<a href="mailto:vanshdwivedi05@gmail.com">
vanshdwivedi05@gmail.com
</a>
</div>
""", unsafe_allow_html=True) 