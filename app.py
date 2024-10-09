import streamlit as st
import os
from PIL import Image

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set page config
st.set_page_config(
    page_title="Dr. Spanos EDS Chatbot",
    page_icon=os.path.join(current_dir, "assets", "favicon.ico"),
    layout="centered"
)

# Load images
avatar_doctor = Image.open(os.path.join(current_dir, "assets", "AvatarDoctor.svg"))
avatar_zebra = Image.open(os.path.join(current_dir, "assets", "AvatarZebra.svg"))
disclaimer_image = Image.open(os.path.join(current_dir, "assets", "disclaimer.png"))

# Custom CSS to match your design
st.markdown("""
<style>
    .stApp {
        background-color: #f0f0f0;
    }
    .stTitle {
        color: #D9376E !important;
        font-size: 2.5rem !important;
    }
    .stSubheader {
        color: #262730 !important;
        font-size: 1rem !important;
    }
    .stChatInputContainer {
        border: 2px solid #FF8E3C !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background-color: #FF8E3C !important;
        color: white !important;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Dr. Spanos EDS Chatbot")
st.subheader("This Chatbot is specifically only trained on Dr. Spanos Research © 2024")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_doctor if message["role"] == "assistant" else avatar_zebra):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar=avatar_zebra):
        st.markdown(prompt)

    # Get bot response (replace this with your actual chatbot logic)
    response = "This is a placeholder response. Replace this with your actual chatbot logic."
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with st.chat_message("assistant", avatar=avatar_doctor):
        st.markdown(response)

# Display disclaimer
st.image(disclaimer_image, use_column_width=True)
st.markdown("This Chatbot is built by BethCNC © 2024")