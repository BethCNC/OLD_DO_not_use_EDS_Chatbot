import React, { useState, useEffect } from 'react';
import './App.css';
import Chatbot from './components/Chatbot';
import { initVectorStore } from './utils/langchain';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initialize = async () => {
      try {
        await initVectorStore();
        setIsLoading(false);
      } catch (err) {
        console.error("Failed to initialize:", err);
        setError(err.message || "Failed to connect to the database. Please try again later.");
        setIsLoading(false);
      }
    };

    initialize();
  }, []);

  if (isLoading) {
    return <div>Loading... Please wait while we connect to the database.</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Chatbot</h1>
      </header>
      <main>
        <Chatbot />
      </main>
    </div>
  );
}

export default App;