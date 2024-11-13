import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import './App.css';
import Login from './Login';

function ChatApp({ isLoggedIn, onSignOut }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isChatStarted, setIsChatStarted] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSend = () => {
    if (input.trim()) {
      if (!isChatStarted) {
        setIsChatStarted(true);
      }


      setMessages([...messages, { sender: 'user', text: input }]);

      setIsLoading(true);

      // bot answer -> 기다림
      setTimeout(() => {
        setMessages(prevMessages => [
          ...prevMessages,
          { sender: 'bot', text: '안녕하세요! 만나서 반갑습니다. 어떻게 도와드릴까요?' }
        ]);
        setIsLoading(false);
      }, 2000); // gpt 처럼 좀 기다렸다 나오는 

      setInput('');
    }
  };

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen); // dropdown 관련
  };

  return (
    <div className="chat-container">
      <div className="header">
        <img src="your-logo.png" alt="Logo" className="logo" />
        <span className="title">NoticeHub</span>
        {!isLoggedIn ? (
          <button
            className="sign-in-button"
            onClick={() => navigate('/login')} // login 중
          >
            Sign in
          </button>
        ) : (
          <div className="user-icon-container">
            <img
              src="user-icon.png" // 사용자 이미지 icon 수정 예정
              alt="User Icon"
              className="user-icon"
              onClick={toggleDropdown} // click 시 dropdown
            />
            {dropdownOpen && (
              <div className="dropdown-menu">
                <button onClick={onSignOut}>Sign out</button>
                <button>구독 정보</button>
              </div>
            )}
          </div>
        )}
      </div>

      {!isChatStarted && (
        <div className="initial-prompt">
          <h2>도움이 필요하시면 "도움!" 을 외치세요</h2>
          <div className="input-container">
            <input
              type="text"
              placeholder="메시지 입력"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="input-box"
            />
            <button onClick={handleSend} className="send-button">→</button>
          </div>
        </div>
      )}

      {isChatStarted && (
        <div className="chat-box">
          {messages.map((message, index) => (
            <div
              key={index}
              className={message.sender === 'user' ? 'user-message' : 'bot-message'}
            >
              {message.sender === 'bot' && (
                <img
                  src="default-icon.png" // 수정 예정 : icon 추가 예정 (gpt icon)
                  alt="Chatbot Icon"
                  className="bot-icon"
                />
              )}
              <span>{message.text}</span>
            </div>
          ))}
          
          {/* 로딩 중일 때 메시지 표시 */}
          {isLoading && (
            <div className="message bot-message">
              <img
                src="default-icon.png" // 수정 예정 : icon 추가 
                alt="Chatbot Icon"
                className="bot-icon"
              />
              <span>Notice Bot is now thinking...</span>
            </div>
          )}

          <div className="input-container">
            <input
              type="text"
              placeholder="메시지 입력"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="input-box"
            />
            <button onClick={handleSend} className="send-button">→</button>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  const handleSignIn = () => {
    setIsLoggedIn(true);
  };

  const handleSignOut = () => {
    setIsLoggedIn(false);
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={<ChatApp isLoggedIn={isLoggedIn} onSignOut={handleSignOut} />} />
        <Route path="/login" element={<Login onSignIn={handleSignIn} />} />
      </Routes>
    </Router>
  );
}

export default App;
