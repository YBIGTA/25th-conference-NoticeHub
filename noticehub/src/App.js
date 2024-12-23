import React, { useState, useEffect } from "react";
import "./App.css";
import "./index.css";
import { v4 as uuidv4 } from "uuid";

function App() {
  const [showChatPage, setShowChatPage] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  const handleStartClick = () => {
    setFadeOut(true);
    setTimeout(() => {
      setShowChatPage(true);
      setFadeOut(false);
    }, 1000);
  };

  return (
    <div className={`container ${fadeOut ? "fade-out" : "fade-in"}`}>
      {!showChatPage ? (
        <div className="card">
          <div className="avatar">
            <img
              src="https://static.vecteezy.com/system/resources/previews/040/532/375/non_2x/cute-eagle-wearing-tie-and-glasses-cartoon-icon-illustration-animal-education-icon-concept-isolated-premium-flat-cartoon-style-vector.jpg"
              alt="Robot Avatar"
              className="avatar-img"
            />
          </div>
          <h1 className="title">Notice Hub</h1>
          <p className="description">
            If you have any questions, <br />
            Start chatting!
          </p>
          <button className="start-btn" onClick={handleStartClick}>
            Start now
          </button>
        </div>
      ) : (
        <ChatPage />
      )}
    </div>
  );
}

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("");

  // 세션 ID를 생성하고 설정
  useEffect(() => {
    const id = uuidv4();
    setSessionId(id);
    console.log("Generated Session ID:", id); // 디버깅용
  }, []);

  const handleInputChange = (e) => {
    setInput(e.target.value);
    const target = e.target;
    target.style.height = "auto";
    target.style.height = `${Math.min(target.scrollHeight, 150)}px`;
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://43.201.55.1:8080/request_rag", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          session_id: sessionId, // 동적으로 생성된 세션 ID 사용
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `Failed to fetch bot response. Status: ${response.status}, Error: ${errorText}`
        );
      }

      const data = await response.json();
      const botMessage = { sender: "bot", text: data.reply };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Fetch error details:", error);
      const errorMessage = {
        sender: "bot",
        text: `Sorry, something went wrong. Error details: ${error.message}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="chat-container">
      <h1 className="chat-title">Notice HUB</h1>
      <div className="chat-box">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${
                msg.sender === "user" ? "user-message" : "bot-message"
              }`}
            >
              {msg.sender === "bot" && (
                <div className="bot-message-icon">
                  <img
                    src="https://static.vecteezy.com/system/resources/previews/040/532/375/non_2x/cute-eagle-wearing-tie-and-glasses-cartoon-icon-illustration-animal-education-icon-concept-isolated-premium-flat-cartoon-style-vector.jpg"
                    alt="Bot Icon"
                  />
                </div>
              )}
              <div className="message-text">{msg.text}</div>
            </div>
          ))}
        </div>
        <div className="chat-input-container">
          <textarea
            className="chat-input"
            placeholder="Type your message..."
            value={input}
            onChange={handleInputChange}
            rows={1}
            style={{ resize: "none", overflowY: "auto" }}
          />
          <button className="send-btn" onClick={handleSend}></button>
        </div>
      </div>
    </div>
  );
}

export default App;
