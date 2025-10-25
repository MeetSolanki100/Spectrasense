import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Trash2, RefreshCw, MessageSquare, Settings, Database, Camera, Upload, Eye, UserPlus, Users } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';
const FACE_REC_API = 'http://localhost:5006';

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
  
  // Vision-related states
  const [visionStatus, setVisionStatus] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [realtimeAnalysis, setRealtimeAnalysis] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);
  const analysisIntervalRef = useRef(null);

  // Face Recognition states
  const [faceRecMonitoring, setFaceRecMonitoring] = useState(false);
  const [latestIdentification, setLatestIdentification] = useState(null);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [newPersonName, setNewPersonName] = useState('');
  const [videoFeedUrl, setVideoFeedUrl] = useState('');
  const identificationIntervalRef = useRef(null);

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

  // Vision Encoder Functions
  const fetchVisionStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/vision/status`);
      const data = await response.json();
      setVisionStatus(data);
    } catch (error) {
      console.error('Failed to fetch vision status:', error);
    }
  };

  const initializeVisionEncoder = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/vision/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      if (data.status === 'success') {
        alert('Vision Encoder initialized successfully!');
        await fetchVisionStatus();
      }
    } catch (error) {
      console.error('Failed to initialize vision encoder:', error);
      alert('Failed to initialize vision encoder');
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
      }
    } catch (error) {
      console.error('Failed to start camera:', error);
      alert('Failed to access camera. Please ensure camera permissions are granted.');
    }
  };

  const toggleRealtimeAnalysis = async () => {
    if (!realtimeAnalysis) {
      // Check if vision encoder is initialized before starting
      if (!visionStatus || !visionStatus.models_loaded) {
        alert('Initializing Vision Encoder... Please wait.');
        await initializeVisionEncoder();
        // Fetch updated status after initialization
        await fetchVisionStatus();
      }
      // Only toggle if initialization was successful or already initialized
      const statusCheck = await fetch(`${API_BASE_URL}/api/vision/status`);
      const statusData = await statusCheck.json();
      if (!statusData.models_loaded) {
        alert('Vision Encoder initialization failed. Please try again.');
        return;
      }
    }
    setRealtimeAnalysis(!realtimeAnalysis);
  };

  // Effect to handle real-time analysis
  useEffect(() => {
    console.log('Real-time analysis effect triggered:', { realtimeAnalysis, cameraActive });
    
    if (realtimeAnalysis && cameraActive) {
      console.log('Starting real-time analysis interval...');
      // Start continuous analysis
      analysisIntervalRef.current = setInterval(() => {
        console.log('Capturing and analyzing frame...');
        captureAndAnalyzeRealtime();
      }, 2000); // Analyze every 2 seconds (increased from 1s for better performance)
    } else {
      // Stop continuous analysis
      if (analysisIntervalRef.current) {
        console.log('Stopping real-time analysis interval');
        clearInterval(analysisIntervalRef.current);
        analysisIntervalRef.current = null;
      }
    }

    return () => {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
      }
    };
  }, [realtimeAnalysis, cameraActive]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setRealtimeAnalysis(false);
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    setLoading(true);
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];
      
      const response = await fetch(`${API_BASE_URL}/api/vision/analyze-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Image })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setAnalysisResult(data.analysis);
      }
    } catch (error) {
      console.error('Failed to analyze frame:', error);
      alert('Failed to analyze frame');
    } finally {
      setLoading(false);
    }
  };

  const captureAndAnalyzeRealtime = async () => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('Video or canvas ref not available');
      return;
    }
    
    // Check if video is ready
    if (videoRef.current.readyState !== videoRef.current.HAVE_ENOUGH_DATA) {
      console.log('Video not ready yet');
      return;
    }
    
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
      
      const base64Image = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]; // Lower quality for speed
      
      console.log('Sending frame for analysis...');
      const response = await fetch(`${API_BASE_URL}/api/vision/analyze-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Image })
      });
      
      console.log('Analysis response status:', response.status);
      
      // Handle 503 Service Unavailable (Vision Encoder not initialized)
      if (response.status === 503) {
        console.warn('Vision Encoder not initialized, stopping real-time analysis');
        setRealtimeAnalysis(false);
        alert('Vision Encoder is not initialized. Please initialize it first.');
        return;
      }
      
      const data = await response.json();
      console.log('Analysis result:', data);
      if (data.status === 'success') {
        setAnalysisResult(data.analysis);
      }
    } catch (error) {
      console.error('Real-time analysis error:', error);
      // Don't show alert for real-time errors to avoid spam
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/api/vision/analyze`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        setAnalysisResult(data.analysis);
        setUploadedImage(`data:image/jpeg;base64,${data.images.original}`);
        setAnnotatedImage(`data:image/jpeg;base64,${data.images.annotated}`);
      }
    } catch (error) {
      console.error('Failed to analyze image:', error);
      alert('Failed to analyze image');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'vision') {
      fetchVisionStatus();
    }
  }, [activeTab]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Face Recognition Functions
  const startFaceRecMonitoring = async () => {
    try {
      console.log('Attempting to start face recognition monitoring...');
      console.log('Connecting to:', `${FACE_REC_API}/start_monitoring`);
      
      const response = await fetch(`${FACE_REC_API}/start_monitoring`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.text();
      console.log('Start monitoring response:', result);
      
      setFaceRecMonitoring(true);
      
      // Set video feed URL - no timestamp needed for MJPEG stream
      // The browser will handle the continuous multipart stream automatically
      setVideoFeedUrl(`${FACE_REC_API}/video_feed`);
      console.log('Video feed URL set:', `${FACE_REC_API}/video_feed`);
      
      // Poll for identification updates every 2 seconds
      identificationIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${FACE_REC_API}/latest_identification`);
          const data = await response.json();
          setLatestIdentification(data);
        } catch (error) {
          console.error('Failed to fetch identification:', error);
        }
      }, 2000);
      
      console.log('Face recognition monitoring started successfully!');
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      alert(`Failed to start face recognition monitoring.\n\nError: ${error.message}\n\nMake sure the Face Recognition backend is running on port 5006.\n\nIf Vision Encoder is using the camera, stop it first.`);
    }
  };

  const stopFaceRecMonitoring = async () => {
    try {
      await fetch(`${FACE_REC_API}/stop_monitoring`);
      setFaceRecMonitoring(false);
      setVideoFeedUrl('');
      
      if (identificationIntervalRef.current) {
        clearInterval(identificationIntervalRef.current);
        identificationIntervalRef.current = null;
      }
      setLatestIdentification(null);
    } catch (error) {
      console.error('Failed to stop monitoring:', error);
    }
  };

  const registerNewFace = async () => {
    if (!newPersonName.trim()) {
      alert('Please enter a name');
      return;
    }

    try {
      const response = await fetch(`${FACE_REC_API}/register_face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newPersonName })
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        alert(`Successfully registered ${newPersonName}!`);
        setShowRegisterModal(false);
        setNewPersonName('');
      } else {
        alert(data.message || 'Failed to register face');
      }
    } catch (error) {
      console.error('Failed to register face:', error);
      alert('Failed to register face');
    }
  };

  const clearKnownFaces = async () => {
    if (!confirm('Are you sure you want to clear all known faces? This cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`${FACE_REC_API}/clear_faces`, {
        method: 'POST'
      });
      
      const data = await response.json();
      if (data.status === 'success') {
        alert('All known faces cleared successfully');
        setLatestIdentification(null);
      }
    } catch (error) {
      console.error('Failed to clear faces:', error);
      alert('Failed to clear known faces');
    }
  };

  useEffect(() => {
    return () => {
      if (identificationIntervalRef.current) {
        clearInterval(identificationIntervalRef.current);
      }
    };
  }, []);

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
            onClick={() => setActiveTab('vision')}
            className={`flex-1 py-2 px-4 rounded-lg transition-all ${
              activeTab === 'vision'
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Eye className="w-4 h-4 inline mr-2" />
            Vision
          </button>
          <button
            onClick={() => setActiveTab('faceRec')}
            className={`flex-1 py-2 px-4 rounded-lg transition-all ${
              activeTab === 'faceRec'
                ? 'bg-purple-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Users className="w-4 h-4 inline mr-2" />
            Face Rec
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

        {activeTab === 'vision' && (
          <div className="space-y-6">
            {/* Vision Status */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Vision Encoder</h2>
                <div className="flex items-center space-x-2">
                  {visionStatus && (
                    <div className={`px-3 py-1 rounded-full text-sm ${
                      visionStatus.models_loaded ? 'bg-green-500' : 'bg-yellow-500'
                    } bg-opacity-20`}>
                      {visionStatus.models_loaded ? '● Models Loaded' : '● Not Initialized'}
                    </div>
                  )}
                  <button
                    onClick={initializeVisionEncoder}
                    disabled={loading || (visionStatus && visionStatus.models_loaded)}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Initialize Models
                  </button>
                </div>
              </div>
              {visionStatus && (
                <p className="text-sm text-gray-400">Device: {visionStatus.device}</p>
              )}
            </div>

            {/* Camera Section */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
              <h3 className="text-lg font-bold mb-4">Real-Time Camera Analysis</h3>
              
              <div className="space-y-4">
                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={cameraActive ? stopCamera : startCamera}
                    className={`px-4 py-2 rounded-lg transition-all ${
                      cameraActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    <Camera className="w-4 h-4 inline mr-2" />
                    {cameraActive ? 'Stop Camera' : 'Start Camera'}
                  </button>
                  
                  {cameraActive && (
                    <>
                      <button
                        onClick={toggleRealtimeAnalysis}
                        className={`px-4 py-2 rounded-lg transition-all ${
                          realtimeAnalysis 
                            ? 'bg-orange-600 hover:bg-orange-700' 
                            : 'bg-blue-600 hover:bg-blue-700'
                        }`}
                      >
                        <RefreshCw className={`w-4 h-4 inline mr-2 ${realtimeAnalysis ? 'animate-spin' : ''}`} />
                        {realtimeAnalysis ? 'Stop Real-Time' : 'Start Real-Time'}
                      </button>
                      
                      {!realtimeAnalysis && (
                        <button
                          onClick={captureAndAnalyze}
                          disabled={loading}
                          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-all disabled:opacity-50"
                        >
                          {loading ? <RefreshCw className="w-4 h-4 inline mr-2 animate-spin" /> : <Eye className="w-4 h-4 inline mr-2" />}
                          Analyze Once
                        </button>
                      )}
                    </>
                  )}
                </div>
                
                {realtimeAnalysis && (
                  <div className="bg-blue-600 bg-opacity-20 rounded-lg p-3 border border-blue-500 border-opacity-30">
                    <p className="text-sm text-blue-300">
                      <RefreshCw className="w-4 h-4 inline mr-2 animate-spin" />
                      Real-time analysis active - Analyzing every second
                    </p>
                  </div>
                )}

                <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ minHeight: '400px' }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-auto"
                    style={{ display: cameraActive ? 'block' : 'none' }}
                  />
                  {!cameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                      <div className="text-center">
                        <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                        <p>Camera not active</p>
                      </div>
                    </div>
                  )}
                </div>
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
            </div>

            {/* Upload Section */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
              <h3 className="text-lg font-bold mb-4">Upload Image for Analysis</h3>
              
              <div className="space-y-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-all disabled:opacity-50"
                >
                  <Upload className="w-4 h-4 inline mr-2" />
                  Upload Image
                </button>

                {(uploadedImage || annotatedImage) && (
                  <div className="grid grid-cols-2 gap-4">
                    {uploadedImage && (
                      <div>
                        <p className="text-sm text-gray-400 mb-2">Original</p>
                        <img src={uploadedImage} alt="Original" className="w-full rounded-lg" />
                      </div>
                    )}
                    {annotatedImage && (
                      <div>
                        <p className="text-sm text-gray-400 mb-2">Annotated</p>
                        <img src={annotatedImage} alt="Annotated" className="w-full rounded-lg" />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Analysis Results */}
            {analysisResult && (
              <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
                <h3 className="text-lg font-bold mb-4">Analysis Results</h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-purple-400 mb-2">Room Description</h4>
                    <p className="text-gray-300">{analysisResult.room_description}</p>
                  </div>

                  {analysisResult.objects && analysisResult.objects.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-green-400 mb-2">Detected Objects</h4>
                      <div className="flex flex-wrap gap-2">
                        {analysisResult.objects.map((obj, idx) => (
                          <span key={idx} className="px-3 py-1 bg-green-600 bg-opacity-30 rounded-full text-sm">
                            {typeof obj === 'string' ? obj : `${obj.class} (${(obj.confidence * 100).toFixed(0)}%)`}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {analysisResult.text_blocks && analysisResult.text_blocks.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-yellow-400 mb-2">Detected Text</h4>
                      <div className="space-y-2">
                        {analysisResult.text_blocks.map((block, idx) => (
                          <div key={idx} className="bg-yellow-600 bg-opacity-20 rounded-lg p-3">
                            <p className="font-mono">
                              {typeof block === 'string' ? block : block.text}
                              {block.confidence && (
                                <span className="text-xs text-gray-400 ml-2">
                                  ({(block.confidence * 100).toFixed(0)}%)
                                </span>
                              )}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {analysisResult.handwritten_text && (
                    <div>
                      <h4 className="font-semibold text-blue-400 mb-2">Handwritten Text</h4>
                      <div className="bg-blue-600 bg-opacity-20 rounded-lg p-3">
                        <p className="font-mono">{analysisResult.handwritten_text}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'faceRec' && (
          <div className="space-y-6">
            {/* Face Recognition Control Panel */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold">Face Recognition Monitor</h2>
                <div className={`px-3 py-1 rounded-full text-sm ${
                  faceRecMonitoring ? 'bg-green-500 bg-opacity-20 text-green-400' : 'bg-gray-500 bg-opacity-20 text-gray-400'
                }`}>
                  {faceRecMonitoring ? '● Active' : '● Inactive'}
                </div>
              </div>

              <div className="flex flex-wrap gap-3 mb-6">
                <button
                  onClick={faceRecMonitoring ? stopFaceRecMonitoring : startFaceRecMonitoring}
                  className={`px-4 py-2 rounded-lg transition-all ${
                    faceRecMonitoring ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <Users className="w-4 h-4 inline mr-2" />
                  {faceRecMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
                </button>

                <button
                  onClick={clearKnownFaces}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-all"
                >
                  <Trash2 className="w-4 h-4 inline mr-2" />
                  Clear Known Faces
                </button>
              </div>

              {/* Video Feed */}
              <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ minHeight: '480px' }}>
                {faceRecMonitoring && videoFeedUrl ? (
                  <img
                    src={videoFeedUrl}
                    alt="Face Recognition Feed"
                    className="w-full h-auto"
                    style={{ maxHeight: '640px', objectFit: 'contain' }}
                    onError={(e) => {
                      console.error('Video feed error - check if camera is available');
                      console.error('Make sure Vision Encoder camera is stopped');
                    }}
                    onLoad={() => {
                      console.log('Video feed stream connected successfully');
                    }}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Users className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Face Recognition Inactive</p>
                      <p className="text-sm mt-2">Click "Start Monitoring" to begin</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Identification Info */}
            {latestIdentification && latestIdentification.name && (
              <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
                <h3 className="text-lg font-bold mb-4">Latest Identification</h3>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Name:</span>
                    <div className="flex items-center gap-2">
                      <span className={`font-bold ${
                        latestIdentification.is_unknown ? 'text-yellow-400' : 'text-green-400'
                      }`}>
                        {latestIdentification.name}
                      </span>
                      {latestIdentification.is_unknown && (
                        <button
                          onClick={() => setShowRegisterModal(true)}
                          className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm transition-all"
                        >
                          <UserPlus className="w-3 h-3 inline mr-1" />
                          Register
                        </button>
                      )}
                    </div>
                  </div>

                  {latestIdentification.time && (
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Time:</span>
                      <span className="font-mono">{latestIdentification.time}</span>
                    </div>
                  )}

                  {latestIdentification.location && (
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Location:</span>
                      <span>{latestIdentification.location}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Info Panel */}
            <div className="bg-black bg-opacity-30 backdrop-blur-lg rounded-xl p-6 border border-purple-500 border-opacity-30">
              <h3 className="text-lg font-bold mb-4">How It Works</h3>
              
              <div className="space-y-3 text-sm text-gray-300">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                  <div>
                    <strong className="text-green-400">Green Box:</strong> Known person identified
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2"></div>
                  <div>
                    <strong className="text-yellow-400">Yellow Box:</strong> Unknown person detected - click Register to add them
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <div>
                    <strong className="text-red-400">No Box:</strong> Anti-spoofing detected fake face (photo/video)
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Register Face Modal */}
        {showRegisterModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-900 rounded-xl p-6 max-w-md w-full mx-4 border border-purple-500">
              <h3 className="text-xl font-bold mb-4">Register New Face</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Person's Name</label>
                  <input
                    type="text"
                    value={newPersonName}
                    onChange={(e) => setNewPersonName(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && registerNewFace()}
                    placeholder="Enter name"
                    className="w-full bg-gray-800 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    autoFocus
                  />
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={registerNewFace}
                    className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-all"
                  >
                    Register
                  </button>
                  <button
                    onClick={() => {
                      setShowRegisterModal(false);
                      setNewPersonName('');
                    }}
                    className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
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