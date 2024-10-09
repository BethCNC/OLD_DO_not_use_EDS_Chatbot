import streamlit as st
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Dr. Spanos EDS Chatbot",
    page_icon="assets/favicon.ico",
    layout="centered"
)

# Load images
avatar_doctor = Image.open("assets/AvatarDoctor.png")
avatar_zebra = Image.open("assets/AvatarZebra.png")
disclaimer_image = Image.open("assets/disclaimer.png")

# Custom CSS to match your design
st.markdown("""
<style>
    .stApp {
        background-color: #f0f0f0;
    }
    .stTitle {
        color: #D9376E;
        font-size: 2.5rem;
    }
    .stSubheader {
        color: #262730;
        font-size: 1rem;
    }
    .stTextInput {
        border: 2px solid #FF8E3C;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #FF8E3C;
        color: white;
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
    # Display user message in chat message container
    with st.chat_message("user", avatar=avatar_zebra):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response = "This is a placeholder response. Replace this with your actual chatbot logic."
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=avatar_doctor):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display disclaimer
st.image(disclaimer_image, use_column_width=True)
st.markdown("This Chatbot is built by BethCNC © 2024")