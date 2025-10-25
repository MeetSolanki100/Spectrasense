import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Trash2, RefreshCw, MessageSquare, Settings, Database } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export default function App() {
  const [chats, setChats] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [stats, setStats] = useState(null);
  const [translate, setTranslate] = useState(false);
  const [targetLang, setTargetLang] = useState('hi');
  const [ws, setWs] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchChats();
    fetchStats();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chats]);

  const fetchChats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/chats?limit=100`);
      const data = await response.json();
      if (data.status === 'success') {
        setChats(data.chats);
      }
    } catch (error) {
      console.error('Failed to fetch chats:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`);
      const data = await response.json();
      if (data.status === 'success') {
        setStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          message: currentMessage,
          translate: translate,
          target_lang: targetLang
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setCurrentMessage('');
        await fetchChats();
        await fetchStats();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      alert('Failed to send message. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const startVoiceRecording = async () => {
    setRecording(true);
    
    try {
      const socket = new WebSocket(`ws://localhost:8000/ws/voice`);
      
      socket.onopen = () => {
        socket.send(JSON.stringify({
          action: 'record',
          duration: 5,
          translate: translate,
          target_lang: targetLang
        }));
      };

      socket.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        
        setRecording(false);
        socket.close();
        
        if (data.status === 'success') {
          // Multiple refresh attempts to ensure UI updates
          await fetchChats();
          await fetchStats();
          
          // Additional refresh after delay
          setTimeout(() => {
            fetchChats();
            fetchStats();
          }, 1000);
        } else if (data.status === 'no_speech') {
          alert('No speech detected. Please try again.');
        } else if (data.error) {
          alert(`Error: ${data.error}`);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        alert('Voice recording failed. Make sure the backend is running.');
        setRecording(false);
      };

      setWs(socket);
    } catch (error) {
      console.error('Failed to start recording:', error);
      setRecording(false);
    }
  };

  const deleteChat = async (chatId) => {
    if (!confirm('Are you sure you want to delete this chat?')) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/delete`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_ids: [chatId] })
      });

      const data = await response.json();
      if (data.status === 'success') {
        await fetchChats();
        await fetchStats();
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
      alert('Failed to delete chat');
    }
  };

  const clearAllChats = async () => {
    if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/chats/clear`, {
        method: 'DELETE',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        await fetchChats();
        await fetchStats();
        alert(data.message);
      }
    } catch (error) {
      console.error('Failed to clear chats:', error);
      alert('Failed to clear chats. Check console for details.');
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="bg-black bg-opacity-30 backdrop-blur-lg border-b border-purple-500 border-opacity-30">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <MessageSquare className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Voice Assistant</h1>
                <p className="text-sm text-gray-400">AI-Powered Conversation</p>
              </div>
            </div>
            
            {stats && (
              <div className="flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <Database className="w-4 h-4 text-purple-400" />
                  <span>{stats.total_conversations} chats</span>
                </div>
                <div className={`px-3 py-1 rounded-full ${stats.chatbot_active ? 'bg-green-500' : 'bg-red-500'} bg-opacity-20`}>
                  {stats.chatbot_active ? '● Active' : '● Inactive'}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 pt-6">
        <div className="flex space-x-2 bg-black bg-opacity-30 backdrop-blur-lg rounded-lg p-1">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex-1 py-2 px-4 rounded-lg transition-all ${
              activeTab === 'chat'
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <MessageSquare className="w-4 h-4 inline mr-2" />
            Chat
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`flex-1 py-2 px-4 rounded-lg transition-all ${
              activeTab === 'history'
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Database className="w-4 h-4 inline mr-2" />
            History
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex-1 py-2 px-4 rounded-lg transition-all ${
              activeTab === 'settings'
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Settings className="w-4 h-4 inline mr-2" />
            Settings
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'chat' && (
          <div className="space-y-6">
            {/* Chat Display */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30 h-96 overflow-y-auto">
              {chats.length === 0 ? (
                <div className="h-full flex items-center justify-center text-gray-400">
                  <div className="text-center">
                    <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>No conversations yet. Start chatting!</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {chats.slice(-10).map((chat) => (
                    <div key={chat.id} className="space-y-2">
                      <div className="flex justify-end">
                        <div className="bg-purple-600 bg-opacity-50 rounded-lg px-4 py-2 max-w-md">
                          <p>{chat.user_message}</p>
                        </div>
                      </div>
                      <div className="flex justify-start">
                        <div className="bg-gray-700 bg-opacity-50 rounded-lg px-4 py-2 max-w-md">
                          <p>{chat.bot_response}</p>
                          <p className="text-xs text-gray-400 mt-1">
                            {formatTimestamp(chat.timestamp)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-4 border border-purple-500 border-opacity-30">
              <div className="flex items-center space-x-3">
                <button
                  onClick={startVoiceRecording}
                  disabled={recording || loading}
                  className={`p-3 rounded-lg transition-all ${
                    recording
                      ? 'bg-red-600 animate-pulse'
                      : 'bg-purple-600 hover:bg-purple-700'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <Mic className="w-5 h-5" />
                </button>
                
                <input
                  type="text"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Type your message..."
                  disabled={loading}
                  className="flex-1 bg-gray-800 bg-opacity-50 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                />
                
                <button
                  onClick={sendMessage}
                  disabled={loading || !currentMessage.trim()}
                  className="p-3 rounded-lg bg-purple-600 hover:bg-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>
              
              {recording && (
                <p className="text-center text-sm text-purple-400 mt-2 animate-pulse">
                  🎤 Recording... Speak now!
                </p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold">Chat History</h2>
              <div className="space-x-2">
                <button
                  onClick={fetchChats}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-all"
                >
                  <RefreshCw className="w-4 h-4 inline mr-2" />
                  Refresh
                </button>
                <button
                  onClick={clearAllChats}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-all"
                >
                  <Trash2 className="w-4 h-4 inline mr-2" />
                  Clear All
                </button>
              </div>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {chats.length === 0 ? (
                <p className="text-center text-gray-400 py-8">No chat history</p>
              ) : (
                chats.map((chat) => (
                  <div
                    key={chat.id}
                    className="bg-gray-800 bg-opacity-50 rounded-lg p-4 hover:bg-opacity-70 transition-all"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <p className="font-semibold text-purple-400">User:</p>
                        <p className="text-sm mb-2">{chat.user_message}</p>
                        <p className="font-semibold text-green-400">Assistant:</p>
                        <p className="text-sm">{chat.bot_response}</p>
                      </div>
                      <button
                        onClick={() => deleteChat(chat.id)}
                        className="p-2 hover:bg-red-600 hover:bg-opacity-30 rounded-lg transition-all"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>
                    <p className="text-xs text-gray-500">{formatTimestamp(chat.timestamp)}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
            <h2 className="text-xl font-bold mb-6">Settings</h2>
            
            <div className="space-y-6">
              <div className="space-y-2">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={translate}
                    onChange={(e) => setTranslate(e.target.checked)}
                    className="w-5 h-5 rounded bg-gray-700 border-purple-500"
                  />
                  <span>Enable Translation</span>
                </label>
              </div>

              {translate && (
                <div className="space-y-2">
                  <label className="block text-sm text-gray-400">Target Language</label>
                  <select
                    value={targetLang}
                    onChange={(e) => setTargetLang(e.target.value)}
                    className="w-full bg-gray-800 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="hi">Hindi (हिंदी)</option>
                    <option value="es">Spanish (Español)</option>
                    <option value="fr">French (Français)</option>
                    <option value="de">German (Deutsch)</option>
                    <option value="ja">Japanese (日本語)</option>
                  </select>
                </div>
              )}

              {stats && (
                <div className="bg-gray-800 bg-opacity-50 rounded-lg p-4 space-y-2">
                  <h3 className="font-semibold text-purple-400 mb-3">System Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Total Conversations</p>
                      <p className="text-xl font-bold">{stats.total_conversations}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Status</p>
                      <p className={`text-xl font-bold ${stats.chatbot_active ? 'text-green-400' : 'text-red-400'}`}>
                        {stats.chatbot_active ? 'Active' : 'Inactive'}
                      </p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-gray-400">Database Path</p>
                      <p className="text-xs font-mono bg-gray-900 px-2 py-1 rounded mt-1">
                        {stats.database_path}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}