import { ChatOpenAI } from "@langchain/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { RetrievalQAChain } from "langchain/chains";

const PINECONE_API_KEY = process.env.REACT_APP_PINECONE_API_KEY;
const PINECONE_ENV = process.env.REACT_APP_PINECONE_ENV;
const PINECONE_INDEX = process.env.REACT_APP_PINECONE_INDEX;
const OPENAI_API_KEY = process.env.REACT_APP_OPENAI_API_KEY;

let vectorStore;

const initVectorStore = async () => {
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
};

export const getChatResponse = async (query) => {
  if (!vectorStore) {
    await initVectorStore();
  }

  const model = new ChatOpenAI({ openAIApiKey: OPENAI_API_KEY });
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  const response = await chain.call({
    query: query,
  });

  return response.text;
};