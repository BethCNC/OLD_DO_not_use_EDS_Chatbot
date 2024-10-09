import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone
import tempfile

# OpenAI setup
openai_api_key = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-3.5-turbo-0125",
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

# Initialize Pinecone
pinecone.init(
    api_key=st.secrets["pinecone"]["api_key"],
    environment=st.secrets["pinecone"]["environment"]
)
index_name = st.secrets["pinecone"]["index_name"]

# Initialize Pinecone Vector Store
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Streamlit UI
st.title("Dr. Spanos Chatbot")

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    for file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        # Add to vector store
        vectorstore.add_documents(texts)
        
        # Remove the temporary file
        import os
        os.unlink(temp_file_path)
    
    st.sidebar.success("Files processed and added to the knowledge base!")

# Initialize conversation chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know about Dr. Spanos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        result = conversation_chain({"question": prompt})
        full_response = result['answer']
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})