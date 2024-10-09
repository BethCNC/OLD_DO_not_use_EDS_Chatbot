import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient

# Load environment variables from .env file if running locally
if os.path.exists('.env'):
    load_dotenv()

# Load secrets from streamlit's secrets.toml if deployed
if "pinecone" in st.secrets:
    pinecone_api_key = st.secrets["pinecone"]["api_key"]
    pinecone_index_name = st.secrets["pinecone"]["index_name"]
    openai_api_key = st.secrets["openai"]["api_key"]
else:
    # Fallback to environment variables for local development
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
try:
    pc = PineconeClient(api_key=pinecone_api_key)
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

# Set up OpenAI embeddings
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the avatar images
avatar_doctor = "DrSpanos_Chatbot/assets/AvatarDoctor.svg"
avatar_zebra = "DrSpanos_Chatbot/assets/AvatarZebra.svg"

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