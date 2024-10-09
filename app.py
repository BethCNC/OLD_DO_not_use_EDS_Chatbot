import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient
from PIL import Image

# Load environment variables from .env file
load_dotenv()

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
    except KeyError:
        st.error(f"Configuration for {key} not found in environment or Streamlit secrets.")
        st.stop()

# Get configuration
pinecone_api_key = get_config("PINECONE_API_KEY")
pinecone_index_name = get_config("PINECONE_INDEX_NAME")
openai_api_key = get_config("OPENAI_API_KEY")

# Initialize Pinecone
try:
    pc = PineconeClient(api_key=pinecone_api_key)
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

# Set up OpenAI embeddings
try:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
except Exception as e:
    st.error(f"Error setting up OpenAI embeddings: {str(e)}")
    st.stop()

# Initialize Pinecone vector store
try:
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

# Streamlit UI
st.title("Dr. Spanos Ehler Danlos Syndrome Chatbot")

# Load the avatar images
current_dir = os.path.dirname(os.path.abspath(__file__))
avatar_doctor_path = os.path.join(current_dir, "assets", "AvatarDoctor.png")
avatar_zebra_path = os.path.join(current_dir, "assets", "AvatarZebra.png")

try:
    avatar_doctor = Image.open(avatar_doctor_path)
    avatar_zebra = Image.open(avatar_zebra_path)
except FileNotFoundError as e:
    st.error(f"Error loading avatar images: {str(e)}")
    st.error(f"Current directory: {current_dir}")
    st.error(f"Attempted to load doctor avatar from: {avatar_doctor_path}")
    st.error(f"Attempted to load zebra avatar from: {avatar_zebra_path}")
    avatar_doctor = None
    avatar_zebra = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_zebra if message["role"] == "user" else avatar_doctor):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    with st.chat_message("user", avatar=avatar_zebra):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=avatar_doctor):
        message_placeholder = st.empty()
        full_response = ""

        # Get response from QA chain
        result = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
        response = result['answer']

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})