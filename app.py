import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pinecone

# Initialize Pinecone
pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone_index_name = st.secrets["pinecone"]["index_name"]
pinecone_env_name = st.secrets["pinecone"]["env_name"]

# Debugging: Ensure secrets are correctly loaded (you can remove this in production)
st.write("Pinecone API Key:", pinecone_api_key)
st.write("OpenAI API Key:", st.secrets["openai"]["api_key"])

# Initialize Pinecone environment and client
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env_name)

# Ensure the Pinecone index is created
if pinecone_index_name not in pinecone.list_indexes():
    st.error(f"Pinecone index '{pinecone_index_name}' not found. Please ensure the index exists.")
else:
    index = pinecone.Index(pinecone_index_name)

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])

# Initialize Pinecone vector store
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Initialize OpenAI chat model
llm = ChatOpenAI(
    openai_api_key=st.secrets["openai"]["api_key"],
    model_name="gpt-3.5-turbo",
    temperature=0
)

# Create a conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Streamlit UI
st.title("RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Get response from QA chain
        try:
            result = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
            response = result['answer']
        except Exception as e:
            response = f"An error occurred: {e}"
            st.error(response)

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Debugging: Output secrets in case of issues (you should remove this in production)
print("Secrets:", st.secrets)