import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";

const PINECONE_API_KEY = process.env.REACT_APP_PINECONE_API_KEY;
const PINECONE_ENV = process.env.REACT_APP_PINECONE_ENV;
const PINECONE_INDEX = process.env.REACT_APP_PINECONE_INDEX;
const OPENAI_API_KEY = process.env.REACT_APP_OPENAI_API_KEY;

let vectorStore;
let qa;

const initVectorStore = async () => {
  try {
    console.log("Initializing Pinecone client...");
    const pinecone = new Pinecone({
      apiKey: PINECONE_API_KEY,
      environment: PINECONE_ENV,
    });

    console.log("Getting Pinecone index...");
    const index = pinecone.Index(PINECONE_INDEX);

    console.log("Creating OpenAI embeddings...");
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: OPENAI_API_KEY,
    });

    console.log("Creating vector store...");
    vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });

    console.log("Creating ChatOpenAI model...");
    const model = new ChatOpenAI({
      openAIApiKey: OPENAI_API_KEY,
      modelName: 'gpt-3.5-turbo',
      temperature: 0.0
    });

    console.log("Creating RetrievalQAChain...");
    qa = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    console.log("Vector store initialized successfully.");
  } catch (error) {
    console.error("Error initializing vector store:", error);
    throw new Error("Failed to connect to the database. Please check your credentials and try again.");
  }
};

const getChatResponse = async (question) => {
  if (!qa) {
    throw new Error("Vector store not initialized. Please call initVectorStore first.");
  }

  try {
    console.log("Querying RetrievalQAChain...");
    const response = await qa.call({ query: question });
    console.log("Response received:", response);
    return response.text;
  } catch (error) {
    console.error("Error getting chat response:", error);
    throw new Error("Failed to get a response from the chatbot. Please try again.");
  }
};

export { initVectorStore, getChatResponse };