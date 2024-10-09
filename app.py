import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import asyncio
import joblib

# Load environment variables
load_dotenv()

# OpenAI setup
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-3.5-turbo-0125",
    streaming=True,
)

# Pinecone setup
api_key_pinecone = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key_pinecone)

# Cohere setup
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = CohereClient(api_key=cohere_api_key)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone Vector Store
vectorStore = PineconeVectorStore(index_name="rag-newchatmodel", embedding=embeddings)

# File handling functions
def create_pkl_string(filename):
    file_name, _ = os.path.splitext(filename)
    return f"{file_name}.pkl"

def load_or_parse_data(file_name):
    changed_file_ext = create_pkl_string(file_name)
    data_file = f"data/{changed_file_ext}"
    if os.path.exists(data_file):
        return joblib.load(data_file)
    else:
        # Implement LlamaParse logic here
        # For now, we'll use a placeholder
        return "Parsed data placeholder"

# Vector database creation
def create_vector_database(file_name):
    llama_parse_documents = load_or_parse_data(file_name)
    # Implement the rest of the vector database creation logic
    # ...

# Reranker setup
def reRanker():
    compressor = CohereRerank(client=cohere_client)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorStore.as_retriever(search_kwargs={"k": 5}),
    )

# Streamlit UI
st.title("Chatbot with Document Upload")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("PDF_PATH", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.sidebar.success(f"File {uploaded_file.name} uploaded successfully!")
    asyncio.run(create_vector_database(uploaded_file.name))

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    compression_retriever = reRanker()
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        for chunk in conversational_rag_chain.stream(
            {"input": prompt},
            config={'configurable': {'session_id': 'unique_session_id'}}
        ):
            full_response += chunk
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})