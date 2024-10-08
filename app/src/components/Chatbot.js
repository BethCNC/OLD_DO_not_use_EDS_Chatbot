import React, { useState, useEffect, useRef } from 'react';
import { getChatResponse } from '../utils/langchain';

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { text: input, user: true };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await getChatResponse(input);
      const botMessage = { text: response, user: false };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error getting chat response:', error);
      const errorMessage = { text: 'Sorry, an error occurred. Please try again.', user: false };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbot max-w-md mx-auto p-4 bg-white rounded-lg shadow-md">
      <div className="chat-messages h-96 overflow-y-auto mb-4 p-2 bg-gray-100 rounded">
        {messages.map((message, index) => (
          <div key={index} className={`message p-2 mb-2 rounded ${message.user ? 'bg-blue-500 text-white ml-auto' : 'bg-gray-300 mr-auto'} max-w-[70%]`}>
            {message.text}
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-center items-center">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900"></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-grow p-2 border rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="bg-blue-500 text-white p-2 rounded-r hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        >
          Send
        </button>
      </form>
    </div>
  );
}

export default Chatbot;