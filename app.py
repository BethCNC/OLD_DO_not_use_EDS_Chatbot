import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone as PineconeClient
from io import BytesIO
import base64
from PIL import Image
from typing import List, Dict, Any

# Set page configuration
st.set_page_config(
    page_title="Dr. Spanos EDS Chatbot",
    page_icon="assets/favicon.ico",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS to match your design
st.markdown("""
    <style>
    @font-face {
        font-family: 'MabryPro';
        src: url('app/static/MabryPro-Regular.ttf') format('truetype');
        font-weight: normal;
        font-style: normal;
    }
    
    @font-face {
        font-family: 'MabryPro';
        src: url('app/static/MabryPro-Bold.ttf') format('truetype');
        font-weight: bold;
        font-style: normal;
    }

    @font-face {
        font-family: 'MabryPro';
        src: url('app/static/MabryPro-Black.ttf') format('truetype');
        font-weight: 900;
        font-style: normal;
    }

    html, body, [class*="css"] {
        font-family: 'MabryPro', sans-serif;
    }

    .stApp {
        background-color: #EEEFF2;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }

    .title-container img {
        width: 48px;
        height: 48px;
        margin-right: 1rem;
    }

    h1 {
        color: #FF492F;
        font-family: 'MabryPro', sans-serif;
        font-weight: 900;
        font-size: 48px;
        margin: 0;
    }

    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }

    .stTextInput > div > div > input {
        background-color: #EEEFF2;
        color: #0D0D0D;
        border: 1px solid #2A2A2A;
    }

    .stButton > button {
        background-color: #FE8D3C;
        color: #0D0D0D;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }

    .chat-message.user {
        background-color: #2A2A2A;
        color: #EEEFF2;
    }

    .chat-message.bot {
        background-color: #EEEFF2;
        color: #0D0D0D;
    }

    .chat-message .avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }

    .chat-message .message {
        flex-grow: 1;
    }

    .stWarning {
        background-color: rgba(255, 142, 60, 0.1);
        color: #FF8E3C;
    }

    .disclaimer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(255, 142, 60, 0.1);
        padding: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }

    .disclaimer-icon {
        width: 24px;
        height: 24px;
        flex-shrink: 0;
    }

    .disclaimer-text {
        color: #FF8E3C;
        font-size: 12px;
        line-height: 1.5;
    }

    .input-container {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }

    .input-container .stTextInput {
        flex-grow: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to get configuration
def get_config(key: str) -> str:
    # First, try to get the value from environment variables
    value = os.getenv(key)
    if value:
        return value
    
    # If not found in environment, try to get from Streamlit secrets
    try:
        if key == "PINECONE_API_KEY":
            return st.secrets["pinecone"]["api_key"]
        elif key == "PINECONE_INDEX_NAME":
            return st.secrets["pinecone"]["index_name"]
        elif key == "OPENAI_API_KEY":
            return st.secrets["openai"]["api_key"]
    except KeyError:
        st.error(f"Configuration for {key} not found in environment or Streamlit secrets.")
        st.stop()
    
    return ""

# Get configuration
pinecone_api_key = get_config("PINECONE_API_KEY")
pinecone_index_name = get_config("PINECONE_INDEX_NAME")
openai_api_key = get_config("OPENAI_API_KEY")

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize Pinecone
pc = PineconeClient(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Create the vector store
vectorstore = Pinecone.from_existing_index(index_name=pinecone_index_name, embedding=embeddings)

# Debug: Check retrieved documents (only in development)
def debug_retriever(query: str):
    if os.getenv('STREAMLIT_ENV') == 'development':
        docs = vectorstore.similarity_search(query, k=3)
        st.write("Debug: Retrieved Documents")
        for i, doc in enumerate(docs):
            st.write(f"Document {i + 1}:")
            st.write(doc.page_content[:200] + "...")  # Print first 200 characters

# Initialize OpenAI chat model
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

# Create a ConversationBufferMemory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Function to convert image to base64
def img_to_base64(img: Image.Image) -> str:
    if img is not None:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return ""

# Streamlit UI
current_dir = os.path.dirname(os.path.abspath(__file__))
favicon_path = os.path.join(current_dir, "assets", "favicon.ico")
favicon_base64 = img_to_base64(Image.open(favicon_path))

st.markdown(f"""
    <div class="title-container">
        <img src="data:image/png;base64,{favicon_base64}" alt="Favicon" />
        <h1>Dr. Spanos Ehler Danlos Syndrome Chatbot</h1>
    </div>
""", unsafe_allow_html=True)

# Load the avatar and disclaimer images
avatar_doctor_path = os.path.join(current_dir, "assets", "AvatarDoctor.png")
avatar_zebra_path = os.path.join(current_dir, "assets", "AvatarZebra.png")
disclaimer_icon_path = os.path.join(current_dir, "assets", "Disclaimer.png")

try:
    avatar_doctor = Image.open(avatar_doctor_path)
    avatar_zebra = Image.open(avatar_zebra_path)
    disclaimer_icon = Image.open(disclaimer_icon_path)
except FileNotFoundError as e:
    st.error(f"Error loading images: {str(e)}")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Attempted to load doctor avatar from: {avatar_doctor_path}")
    st.error(f"Attempted to load zebra avatar from: {avatar_zebra_path}")
    st.error(f"Attempted to load disclaimer icon from: {disclaimer_icon_path}")
    avatar_doctor = avatar_zebra = disclaimer_icon = None

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_zebra if message["role"] == "user" else avatar_doctor):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_zebra):
        st.markdown(prompt)

    # Debug: Check retrieved documents
    debug_retriever(prompt)

    with st.chat_message("assistant", avatar=avatar_doctor):
        message_placeholder = st.empty()
        full_response = ""
        result = qa_chain.invoke({"question": prompt})
        full_response = result['answer']
        message_placeholder.markdown(full_response)

        # Print sources
        st.write("Sources:")
        for i, doc in enumerate(result['source_documents']):
            st.write(f"Source {i + 1}:")
            st.write(doc.page_content[:200] + "...")  # Print first 200 characters

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer
disclaimer_text = "Disclaimer: The information contained on this site and the services provided by Doctor Lee Patient Advocacy are for educational purposes only. Although we have performed extensive research regarding medical conditions, treatments, diagnoses, procedures and medical research, the staff of Doctor Lee Patient Advocacy are not licensed providers of healthcare. The information provided by Doctor Lee Patient Advocacy should not be considered medical advice or used to diagnose or treat any health problem or disease. It is not a substitute for professional care. If you have or suspect you may have a health problem, you should consult your appropriate health care provider."

disclaimer_icon_base64 = img_to_base64(disclaimer_icon) if disclaimer_icon else ""
disclaimer_html = f"""
<div class="disclaimer-container">
    <img src="data:image/png;base64,{disclaimer_icon_base64}" class="disclaimer-icon" />
    <div class="disclaimer-text">{disclaimer_text}</div>
</div>
"""

st.markdown(disclaimer_html, unsafe_allow_html=True)