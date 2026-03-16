import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import uuid
import os

# ─────────────────────────────────────────────
# Page Config  ← MUST be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Visio — AI Caption Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "last_image" not in st.session_state:
    st.session_state.last_image = None

# ─────────────────────────────────────────────
# ALL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

:root{
  --bg:        #080C18;
  --surface:   #0F1628;
  --border:    rgba(201,169,110,0.18);
  --gold:      #C9A96E;
  --gold-dim:  rgba(201,169,110,0.55);
  --pearl:     #F0EDE8;
  --pearl-dim: rgba(240,237,232,0.55);
  --radius:    18px;
  --shadow:    0 24px 64px rgba(0,0,0,0.55);
}

html, body, [class*="css"]{
  font-family: 'DM Sans', sans-serif !important;
}

.stApp{
  background: var(--bg);
  background-image:
    radial-gradient(ellipse 80% 60% at 10% 0%,  rgba(201,169,110,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 90% 100%, rgba(100,120,200,0.07) 0%, transparent 55%);
  min-height: 100vh;
}

.block-container{
  max-width: 1160px !important;
  padding: 2rem 2rem 4rem 2rem !important;
  margin: 0 auto;
}

.site-title{
  font-family: 'Cormorant Garamond', serif;
  font-weight: 300;
  font-size: clamp(48px, 6vw, 76px);
  letter-spacing: 6px;
  color: var(--pearl);
  text-align: center;
  line-height: 1;
}
.site-title span{ color: var(--gold); }

.site-subtitle{
  font-family: 'DM Sans', sans-serif;
  font-weight: 300;
  font-size: 15px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--pearl-dim);
  text-align: center;
  margin-top: 10px;
}

.divider{
  width: 100%;
  height: 1px;
  background: linear-gradient(to right, transparent, var(--gold-dim), transparent);
  margin: 32px 0;
}

.card{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 28px 24px 28px;
  box-shadow: var(--shadow);
}

.section-label{
  font-family: 'DM Sans', sans-serif;
  font-weight: 400;
  font-size: 11px;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-label::after{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

[data-testid="stFileUploader"]{
  background: rgba(201,169,110,0.04) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 14px !important;
  padding: 20px !important;
}
[data-testid="stFileUploaderDropzone"]{ background: transparent !important; }
[data-testid="stFileUploader"] label{ color: var(--pearl-dim) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div span{
  color: var(--pearl-dim) !important;
  font-size: 14px !important;
}

.stButton > button{
  background: linear-gradient(135deg, #C9A96E 0%, #E8C98A 50%, #C9A96E 100%) !important;
  background-size: 200% 200% !important;
  color: #080C18 !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 14px 40px !important;
  cursor: pointer !important;
  transition: all 0.4s ease !important;
  box-shadow: 0 4px 24px rgba(201,169,110,0.35) !important;
  width: 100% !important;
}
.stButton > button:hover{
  box-shadow: 0 8px 36px rgba(201,169,110,0.55) !important;
  transform: translateY(-2px) !important;
}

.caption-result-wrap{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 32px 36px;
  margin-top: 8px;
  position: relative;
  overflow: hidden;
}
.caption-result-wrap::before{
  content: '\201C';
  font-family: 'Cormorant Garamond', serif;
  font-size: 140px;
  color: rgba(201,169,110,0.08);
  position: absolute;
  top: -20px;
  left: 16px;
  line-height: 1;
  pointer-events: none;
}
.caption-eyebrow{
  font-size: 11px;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 14px;
}
.caption-body{
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(22px, 3vw, 32px);
  font-weight: 300;
  color: var(--pearl);
  line-height: 1.55;
  letter-spacing: 0.3px;
}
.caption-empty{
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  color: rgba(240,237,232,0.22);
  letter-spacing: 1px;
}

.about-card{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 30px 36px;
  margin-top: 6px;
}
.about-title{
  font-family: 'Cormorant Garamond', serif;
  font-size: 22px;
  font-weight: 400;
  color: var(--gold);
  letter-spacing: 1px;
  margin-bottom: 14px;
}
.about-body{
  font-size: 14.5px;
  color: var(--pearl-dim);
  line-height: 1.9;
}
.feature-pill{
  display: inline-block;
  background: rgba(201,169,110,0.09);
  border: 1px solid rgba(201,169,110,0.22);
  border-radius: 30px;
  color: var(--gold);
  font-size: 12px;
  letter-spacing: 1.5px;
  padding: 5px 16px;
  margin: 4px 4px 0 0;
  text-transform: uppercase;
}

.footer{
  text-align: center;
  padding-top: 48px;
  color: rgba(240,237,232,0.28);
  font-size: 13px;
  letter-spacing: 0.3px;
  line-height: 2.2;
}
.footer a{
  color: var(--gold-dim) !important;
  text-decoration: none !important;
  border-bottom: 1px solid rgba(201,169,110,0.25);
  padding-bottom: 1px;
}
.footer a:hover{ color: var(--gold) !important; }

img{
  margin: 0 auto !important;
  display: block !important;
  border-radius: 12px;
}
[data-testid="stImage"] img{
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

[data-testid="stAlert"]{
  background: rgba(201,169,110,0.07) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--pearl-dim) !important;
}
div[data-testid="stMarkdownContainer"]{ text-align: center; }
[data-testid="column"]{ text-align: center; }

audio{
  filter: invert(0.9) sepia(0.3) hue-rotate(5deg);
  border-radius: 40px;
  width: 100%;
}
::-webkit-scrollbar{ width: 6px; }
::-webkit-scrollbar-track{ background: var(--bg); }
::-webkit-scrollbar-thumb{ background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGO SVG
# ─────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72" fill="none">
  <circle cx="36" cy="36" r="33" stroke="#C9A96E" stroke-width="1.2" stroke-dasharray="3 4" opacity="0.5"/>
  <circle cx="36" cy="36" r="24" stroke="#C9A96E" stroke-width="1.5" opacity="0.7"/>
  <g transform="translate(36,36)">
    <path d="M0 -14 Q7 -7 14 0 Q7 7 0 14 Q-7 7 -14 0 Q-7 -7 0 -14Z"
          fill="none" stroke="#C9A96E" stroke-width="1.2" opacity="0.5" transform="rotate(0)"/>
    <path d="M0 -14 Q7 -7 14 0 Q7 7 0 14 Q-7 7 -14 0 Q-7 -7 0 -14Z"
          fill="none" stroke="#C9A96E" stroke-width="1.2" opacity="0.5" transform="rotate(60)"/>
    <path d="M0 -14 Q7 -7 14 0 Q7 7 0 14 Q-7 7 -14 0 Q-7 -7 0 -14Z"
          fill="none" stroke="#C9A96E" stroke-width="1.2" opacity="0.5" transform="rotate(120)"/>
    <circle cx="0"   cy="-24" r="2.5" fill="#C9A96E"/>
    <circle cx="24"  cy="0"   r="2.5" fill="#C9A96E"/>
    <circle cx="0"   cy="24"  r="2.5" fill="#C9A96E"/>
    <circle cx="-24" cy="0"   r="2.5" fill="#C9A96E"/>
    <line x1="0" y1="-24" x2="24" y2="0"   stroke="#C9A96E" stroke-width="0.8" opacity="0.4"/>
    <line x1="24" y1="0"  x2="0"  y2="24"  stroke="#C9A96E" stroke-width="0.8" opacity="0.4"/>
    <line x1="0" y1="24"  x2="-24" y2="0"  stroke="#C9A96E" stroke-width="0.8" opacity="0.4"/>
    <line x1="-24" y1="0" x2="0"  y2="-24" stroke="#C9A96E" stroke-width="0.8" opacity="0.4"/>
    <circle cx="0" cy="0" r="6" fill="#C9A96E" opacity="0.9"/>
    <circle cx="0" cy="0" r="3.5" fill="#080C18"/>
    <circle cx="0" cy="0" r="1.5" fill="#C9A96E"/>
  </g>
  <path d="M6 6 L14 6 M6 6 L6 14"   stroke="#C9A96E" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
  <path d="M66 6 L58 6 M66 6 L66 14" stroke="#C9A96E" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
  <path d="M6 66 L14 66 M6 66 L6 58" stroke="#C9A96E" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
  <path d="M66 66 L58 66 M66 66 L66 58" stroke="#C9A96E" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
</svg>
"""

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; margin-bottom:4px;">
  {LOGO_SVG}
</div>
<div class="site-title">VIS<span>I</span>O</div>
<div class="site-subtitle">AI — Powered Image Caption Generator</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Load
# ─────────────────────────────────────────────
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def speak_text(text):
    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
    tts.save(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")
    os.remove(filename)

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip().capitalize()

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.last_image = image
        st.image(image, use_container_width=True)
    elif st.session_state.last_image is not None:
        st.image(st.session_state.last_image, use_container_width=True)
    else:
        st.markdown(
            '<div style="color:rgba(240,237,232,0.2); font-size:13px; '
            'letter-spacing:2px; text-transform:uppercase; padding:40px 0; text-align:center;">'
            '✦  Awaiting Image  ✦</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GENERATE BUTTON
# ─────────────────────────────────────────────
st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    generate_btn = st.button("✦  Generate Caption", use_container_width=True)

if generate_btn and uploaded_file:
    with st.spinner("Reading your image with AI..."):
        st.session_state.caption = generate_caption(st.session_state.last_image)
elif generate_btn and not uploaded_file:
    st.warning("Please upload an image before generating a caption.")

# ─────────────────────────────────────────────
# CAPTION OUTPUT
# ─────────────────────────────────────────────
st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Generated Caption</div>', unsafe_allow_html=True)

if st.session_state.caption:
    st.markdown(f"""
    <div class="caption-result-wrap">
      <div class="caption-eyebrow">AI Vision Analysis</div>
      <div class="caption-body">{st.session_state.caption}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    _, voice_col, _ = st.columns([1.5, 1.5, 1.5])
    with voice_col:
        if st.button("♪  Read Aloud", use_container_width=True):
            speak_text(st.session_state.caption)
else:
    st.markdown("""
    <div class="caption-result-wrap">
      <div class="caption-eyebrow">AI Vision Analysis</div>
      <div class="caption-empty">— awaiting generation —</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────
st.markdown('<div class="divider" style="margin-top:44px;margin-bottom:28px;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ABOUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-card">
  <div class="about-title">How it works</div>
  <div class="about-body">
    Visio leverages a pretrained <strong style="color:#E8C98A;">Vision-Language Transformer</strong>
    (BLIP) to bridge the gap between visual perception and natural language.
    The model interprets spatial relationships, objects, and context within your image
    then distils that understanding into a precise, human-readable caption.
  </div>
  <div style="margin-top: 20px;">
    <span class="feature-pill">Real-time Captioning</span>
    <span class="feature-pill">Voice Synthesis</span>
    <span class="feature-pill">BLIP Transformer</span>
    <span class="feature-pill">Computer Vision</span>
    <span class="feature-pill">NLP</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div style="letter-spacing:4px; font-size:11px; color:rgba(201,169,110,0.4); margin-bottom:12px;">
    ✦  VISIO  ✦
  </div>
  AI Vision-Language Captioning System<br>
  Developed by &nbsp;
  <a href="https://www.linkedin.com/in/vedant-singh-rathore-10aa11377" target="_blank">Vedant Singh Rathore</a>
  &nbsp;&amp;&nbsp;
  <a href="https://www.linkedin.com/in/vansh-dwivedi-45a634349" target="_blank">Vansh Dwivedi</a>
  <br>
  <a href="mailto:ivedantsinghrathore@gmail.com">ivedantsinghrathore@gmail.com</a>
  &nbsp;·&nbsp;
  <a href="mailto:vanshdwivedi05@gmail.com">vanshdwivedi05@gmail.com</a>
  <br><br>
  <span style="font-size:11px; letter-spacing:2px; opacity:0.4;">
    BUILT WITH STREAMLIT &amp; HUGGING FACE TRANSFORMERS
  </span>
</div>
""", unsafe_allow_html=True)