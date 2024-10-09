import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient

# Initialize Pinecone
pc = PineconeClient(api_key=st.secrets.pinecone.api_key)

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.openai.api_key)

# Initialize Pinecone vector store
index_name = st.secrets.pinecone.index_name
index = pc.Index(index_name)
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Initialize OpenAI chat model
llm = ChatOpenAI(
    openai_api_key=st.secrets.openai.api_key,
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
        result = qa_chain({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]})
        response = result['answer']

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Debugging: Output secrets in case of issues (you should remove this in production)
print("Secrets:", st.secrets)