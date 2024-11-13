import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Login({ onSignIn }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = () => {
    // login process 추가 예정 -> 할거면 추가 여기다 하면 되나요? 모르겠어요~
    if (username && password) {
      onSignIn();
      navigate('/'); 
    }
  };

  return (
    <div className="login-container">
      <h2>Login</h2>
      <input
        type="text"
        placeholder="Username"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        className="login-input"
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="login-input"
      />
      <button onClick={handleLogin} className="login-button">Log In</button>
    </div>
  );
}

export default Login;
