import { ChatOpenAI } from "langchain/dist/chat_models/openai";
import { PineconeStore } from "langchain/dist/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/dist/embeddings/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { RetrievalQAChain } from "langchain/dist/chains";

const PINECONE_API_KEY = process.env.REACT_APP_PINECONE_API_KEY;
const PINECONE_ENV = process.env.REACT_APP_PINECONE_ENV;
const PINECONE_INDEX = process.env.REACT_APP_PINECONE_INDEX;
const OPENAI_API_KEY = process.env.REACT_APP_OPENAI_API_KEY;

if (!PINECONE_API_KEY || !PINECONE_ENV || !PINECONE_INDEX || !OPENAI_API_KEY) {
  throw new Error("Missing environment variables");
}

let vectorStore;

const initVectorStore = async () => {
  try {
    const client = new PineconeClient();
    await client.init({
      apiKey: PINECONE_API_KEY,
      environment: PINECONE_ENV,
    });

    const pineconeIndex = client.Index(PINECONE_INDEX);

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: OPENAI_API_KEY,
    });

    vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });
  } catch (error) {
    console.error("Error initializing vector store:", error);
    throw error;
  }
};

export const getChatResponse = async (query) => {
  try {
    if (!vectorStore) {
      await initVectorStore();
    }

    const model = new ChatOpenAI({ openAIApiKey: OPENAI_API_KEY });
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    const response = await chain.call({
      query: query,
    });

    return response.text;
  } catch (error) {
    console.error("Error getting chat response:", error);
    throw error;
  }
};