import React from 'react';

const ChatMessage = ({ message }) => {
  return (
    <div className={`chat-message ${message.user ? 'user' : 'bot'}`}>
      <div className="message-content">{message.text}</div>
    </div>
  );
};

export default ChatMessage;