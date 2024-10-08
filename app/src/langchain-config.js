import { OpenAI } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { PineconeClient } from "@pinecone-database/pinecone";

const OPENAI_API_KEY = process.env.REACT_APP_OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.REACT_APP_PINECONE_API_KEY;
const PINECONE_ENVIRONMENT = process.env.REACT_APP_PINECONE_ENVIRONMENT;
const PINECONE_INDEX = process.env.REACT_APP_PINECONE_INDEX;

export const initLangchain = async () => {
  const client = new PineconeClient();
  await client.init({
    apiKey: PINECONE_API_KEY,
    environment: PINECONE_ENVIRONMENT,
  });

  const pineconeIndex = client.Index(PINECONE_INDEX);

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAI({ openAIApiKey: OPENAI_API_KEY }),
    { pineconeIndex }
  );

  return vectorStore;
};