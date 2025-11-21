import streamlit as st
import pandas as pd
import time
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fake or Nah?",
    page_icon="📰",
    layout="centered"
)

# --- COLOR PALETTE ---
COLOR_BACKGROUND = "#FDF5E6"
COLOR_PRIMARY_TEXT = "#300E0E"
COLOR_BUTTON_TEXT = "#FFFFFF"
COLOR_BUTTON_BACKGROUND = "#F2A1A1"
COLOR_BUTTON_HOVER_TEXT = "#FFE2E2"
COLOR_BUTTON_HOVER = "#F56C42"
COLOR_IMAGE_BORDER = "#000000"

# --- BACKGROUND IMAGE ---
NEWSPAPER_BACKGROUND_BASE64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAgACADASIAAhEBAxEB/8QAFwABAQEBAAAAAAAAAAAAAAAAAAUHBP/EACgQAAIBAgQGAgMBAAAAAAAAAAECAwQFEQYHEiExExQiQVFhcYEykf/EABgBAQEBAQEAAAAAAAAAAAAAAAMFBgcI/8QAHxEBAAEDBAMBAAAAAAAAAAAAAAECAwQREhMhIjFB/9oADAMBAAIRAxEAPwDT5x2hqcizVTVVbJd0hieNqZgGZz3hSAw7b2x841Pxb49z/M62rpKGelpzRpHIpmc7t+7v7wAB9u2NVrM5WvOa/N3rIu9JUTM1cFA3JGI+3p+v7xjR8J8/ocpzdXv3j0lXGkKzKDpjffuDH2A9/bFbr4vLdqvVcY6fDqsVq5TjZWe9rN2yWlNbUV8dTUy0VPTxs5d3lK9w/CgH32wO+J+b1lNRZXDRzSQ01VE8kyxnbuAPSAfYbnE/wAR86p80zjSUUgkpKSPukMg/wAcn7sfYDb6b4E81q1zDLMgq4yD3aaSMkHqCrEH8Y1NNi1TVd13HPHn8m5e/bpxj3e4QzKqq67OK2ermaeVpSCzkknbp+sVzMWJLEkk7knqTj8D1P1j9x09M3MREcNfMzOcr//Z"

# --- CUSTOM STYLING (CSS) ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Playfair+Display:wght@700&display=swap');
        
        /* --- VIBE UPGRADE: TEXTURED BACKGROUND (ROBUST METHOD) --- */
        body {{
            background-image: url("{NEWSPAPER_BACKGROUND_BASE64}");
            background-size: cover;
            background-repeat: repeat;
            background-attachment: fixed; /* Keeps the background still on scroll */
        }}
        
        /* --- VIBE UPGRADE: MAIN CONTENT AS A "PAGE" --- */
        /* This targets the main container where all your content lives */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: {COLOR_BACKGROUND};
            border: 2px solid #A9A9A9;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }}
        
        html, body, [class*="st-"] {{
            font-family: 'Merriweather', serif;
            color: {COLOR_PRIMARY_TEXT};
        }}
        
        h1, h2 {{
            font-family: 'Playfair Display', serif;
            font-weight: 700;
        }}
        
        /* Custom styling for the spinner animation */
        .stSpinner > div > div {{
            border-top-color: {COLOR_BUTTON_HOVER} !important;
            border-right-color: {COLOR_BUTTON_HOVER} !important;
            border-bottom-color: {COLOR_BUTTON_HOVER} !important;
            border-left-color: transparent !important;
            width: 80px !important;
            height: 80px !important;
            border-width: 8px !important;
        }}
        
        .stButton>button {{
            border: 2px solid {COLOR_BUTTON_BACKGROUND};
            border-radius: 5px;
            padding: 10px 20px;
            font-family: 'Playfair Display', serif;
            font-size: 20px;
            font-weight: 700;
            color: {COLOR_BUTTON_BACKGROUND};
            background-color: {COLOR_BUTTON_TEXT};
            transition: all 0.3s ease-in-out;
        }}
        
        .stButton>button:hover {{
            transform: scale(1.05);
            border-color: {COLOR_BUTTON_HOVER};
            background-color: {COLOR_BUTTON_HOVER};
            color: {COLOR_BUTTON_HOVER_TEXT};
        }}
        
        div[data-testid="stImage"] img {{
            border-radius: 10px;
            border: 3px solid {COLOR_IMAGE_BORDER}; 
        }}
    </style>
""", unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = ""
if 'analytics_df' not in st.session_state:
    st.session_state.analytics_df = None

# --- FAKE NEWS DETECTOR SECTION ---
st.image("title_dark.png", use_container_width=True)
st.write("")
st.write("Try our fake news detector by pasting the title & text of the article you want to check:")
title = st.text_input("Insert Article Title...")
article_text = st.text_area("Insert Article Text...", height=200)

st.write("") 
col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    if st.button("CHECK", use_container_width=True):
        if title and article_text:
            with st.spinner("Analyzing..."):
                time.sleep(3) # Simulate your model running.

            import random
            is_true = random.choice([True, False])
            st.session_state.show_results = True
            if is_true:
                st.session_state.prediction = "Our analysis indicates this text is **most likely TRUE!** 👍"
            else:
                st.session_state.prediction = "Our analysis indicates this text is **most likely FAKE!** 👎"
            st.session_state.analytics_df = pd.DataFrame({
                "Metric": ["# of '!?¡' symbols", "# of profanity words", "Sentiment Score"],
                "Value": [random.randint(0, 5), random.randint(0, 3), f"{random.uniform(-1, 1):.2f}"]
            }).set_index("Metric")
        else:
            st.warning("Please fill in both the title and article text.")

# --- RESULTS SECTION ---
if st.session_state.show_results:
    st.header("Results")
    if "TRUE" in st.session_state.prediction:
        st.success(st.session_state.prediction)
    else:
        st.error(st.session_state.prediction)
    st.subheader("Analytics")
    st.write("Key indicators from our analysis:")
    st.table(st.session_state.analytics_df)
    
st.divider()

# --- ABOUT US SECTION ---
st.header("Meet the Team")
st.write("Our goal with this project is to create an accessible tool to combat misinformation and promote media literacy.")
st.write("")
col1, col2 = st.columns([1, 2])
with col1:
    st.image("Nina_Image.jpg", use_container_width=True)
with col2:
    st.subheader("Nina Elmoyan")
    st.write("Nina is the lead data scientist, specializing in Natural Language Processing and model development.")
st.write("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://via.placeholder.com/150", use_container_width=True)
with col2:
    st.subheader("Wynne")
    st.write("Wynne is the project manager and UI/UX designer, ensuring the app is both powerful and user-friendly.")
st.write("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://via.placeholder.com/150", use_container_width=True)
with col2:
    st.subheader("Rhode")
    st.write("Rhode handles the back-end architecture and data engineering, making sure our models run efficiently.")
st.write("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://via.placeholder.com/150", use_container_width=True)
with col2:
    st.subheader("Ramneek")
    st.write("Ramneek is responsible for model validation and quality assurance, rigorously testing the detector's accuracy.")