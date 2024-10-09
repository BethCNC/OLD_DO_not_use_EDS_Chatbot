import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient
from PIL import Image
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Dr. Spanos EDS Chatbot",
    page_icon="DrSpanos_Chatbot/assets/favicon.ico",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS to match your design
st.markdown("""
    <style>
    @font-face {
        font-family: 'MabryPro';
        src: url('DrSpanos_Chatbot/assets/MabryPro-Regular.woff2') format('woff2'),
             url('DrSpanos_Chatbot/assets/MabryPro-Regular.woff') format('woff');
        font-weight: normal;
        font-style: normal;
    }
    
    @font-face {
        font-family: 'MabryPro';
        src: url('DrSpanos_Chatbot/assets/MabryPro-Bold.woff2') format('woff2'),
             url('DrSpanos_Chatbot/assets/MabryPro-Bold.woff') format('woff');
        font-weight: bold;
        font-style: normal;
    }

    @font-face {
        font-family: 'MabryPro';
        src: url('DrSpanos_Chatbot/assets/MabryPro_Black.ttf') format('truetype');
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
def get_config(key):
    # First, try to get the value from environment variables (for local development)
    value = os.getenv(key)
    if value:
        return value
    
    # If not found in environment, try to get from Streamlit secrets (for deployment)
    try:
        if key == "PINECONE_API_KEY":
            return st.secrets["pinecone"]["api_key"]
        elif key == "PINECONE_INDEX_NAME":
            return st.secrets["pinecone"]["index_name"]
        elif key == "OPENAI_API_KEY":
            return st.secrets["openai"]["api_key"]
    except (KeyError, FileNotFoundError):
        st.error(f"Configuration for {key} not found in environment or Streamlit secrets.")
        st.error("Please ensure you have set up your secrets.toml file or environment variables correctly.")
        st.stop()

# Get configuration
pinecone_api_key = get_config("PINECONE_API_KEY")
pinecone_index_name = get_config("PINECONE_INDEX_NAME")
openai_api_key = get_config("OPENAI_API_KEY")

# Set up OpenAI embeddings
try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
except Exception as e:
    st.error(f"Error setting up OpenAI embeddings: {str(e)}")
    st.stop()

# Initialize Pinecone
try:
    pc = PineconeClient(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
except Exception as e:
    st.error(f"Error initializing Pinecone vector store: {str(e)}")
    st.stop()

# Initialize OpenAI chat model
try:
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )
except Exception as e:
    st.error(f"Error initializing OpenAI chat model: {str(e)}")
    st.stop()

# Create a conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Function to convert image to base64
def img_to_base64(img):
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
current_dir = os.path.dirname(os.path.abspath(__file__))
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
    avatar = avatar_zebra if message["role"] == "user" else avatar_doctor
    avatar_base64 = img_to_base64(avatar)
    st.markdown(f"""
        <div class='chat-message {message["role"]}'>
            <img class='avatar' src="data:image/png;base64,{avatar_base64}" />
            <div class='message'>{message["content"]}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Chat input
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
user_input = st.text_input("Type your message here...", key="user_input")
send_button = st.button("Send")
st.markdown("</div>", unsafe_allow_html=True)

if send_button and user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get bot response
    result = qa_chain({"question": user_input, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
    bot_response = result['answer']
    
    # Add bot response to chat
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Clear the input field
    st.session_state.user_input = ""
    
    # Rerun the app to display the new messages
    st.rerun()

# Disclaimer
disclaimer_text = "Disclaimer: The information contained on this site and the services provided by Doctor Lee Patient Advocacy are for educational purposes only. Although we have performed extensive research regarding medical conditions, treatments, diagnoses, procedures and medical research, the staff of Doctor Lee Patient Advocacy are not licensed providers of healthcare. The information provided by Doctor Lee Patient Advocacy should not be considered medical advice or used to diagnose or treat any health problem or disease. It is not a substitute for professional care. If you have or suspect you may have a health problem, you should consult your appropriate health care provider."

disclaimer_icon_base64 = img_to_base64(disclaimer_icon)
disclaimer_html = f"""
<div class="disclaimer-container">
    <img src="data:image/png;base64,{disclaimer_icon_base64}" class="disclaimer-icon" />
    <div class="disclaimer-text">{disclaimer_text}</div>
</div>
"""

st.markdown(disclaimer_html, unsafe_allow_html=True)