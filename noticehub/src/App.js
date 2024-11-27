import React, { useState } from "react";
import "./App.css";

const App = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);

  const handleSend = async () => {
    if (input.trim() === "") return;
    setMessages((prev) => [...prev, { type: "user", text: input }]);
    setInput("");
    setIsThinking(true);

    try {
      const response = await fetch("backend////apiendpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { type: "bot", text: data.answer }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "Bacek" },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">Notice Hub</div>
      </header>

      <div className="chat-view">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${
              msg.type === "user" ? "user-message" : "bot-message"
            }`}
          >
            {msg.text}
          </div>
        ))}
        {isThinking && <div className="bot-message">I am thinking now...</div>}
      </div>

      <div className="input-container chat-input">
        <textarea
          value={input}
          onChange={handleInputChange}
          placeholder="Type your question here"
          rows={1}
        />
        <button onClick={handleSend}>
          <span>↑</span>
        </button>
      </div>
    </div>
  );
};

export default App;
