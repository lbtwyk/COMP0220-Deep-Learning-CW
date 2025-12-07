import React, { useState, useRef, useEffect, useMemo } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

// Word-by-word fade effect with markdown support after completion
function StreamingText({ text, speed = 40 }) {
  const [visibleCount, setVisibleCount] = useState(0)
  const [isComplete, setIsComplete] = useState(false)
  
  const words = React.useMemo(() => {
    return text.split(/(\s+)/).filter(w => w.length > 0)
  }, [text])

  useEffect(() => {
    setVisibleCount(0)
    setIsComplete(false)
  }, [text])

  useEffect(() => {
    if (visibleCount < words.length) {
      const timeout = setTimeout(() => {
        setVisibleCount(prev => prev + 1)
      }, speed)
      return () => clearTimeout(timeout)
    } else if (!isComplete && words.length > 0) {
      setIsComplete(true)
    }
  }, [visibleCount, words.length, speed, isComplete])

  if (isComplete) {
    return (
      <div className="streaming-container complete">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
    )
  }

  return (
    <div className="streaming-container">
      <div className="streaming-words">
        {words.map((word, index) => (
          <span key={index} className={`stream-word ${index < visibleCount ? 'visible' : 'hidden'}`}>
            {word}
          </span>
        ))}
      </div>
      <span className="cursor-blink">|</span>
    </div>
  )
}

// Settings Modal Component
function SettingsModal({ isOpen, onClose, settings, onUpdateSettings }) {
  const [activeTab, setActiveTab] = useState('tts')
  const [localSettings, setLocalSettings] = useState(settings)

  useEffect(() => {
    setLocalSettings(settings)
  }, [settings])

  if (!isOpen) return null

  const handleSave = () => {
    onUpdateSettings(localSettings)
    // Save to localStorage
    localStorage.setItem('signtutor_settings', JSON.stringify(localSettings))
    onClose()
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Settings</h2>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-tabs">
          <button 
            className={`tab ${activeTab === 'tts' ? 'active' : ''}`}
            onClick={() => setActiveTab('tts')}
          >
            🔊 Text-to-Speech
          </button>
          <button 
            className={`tab ${activeTab === 'api' ? 'active' : ''}`}
            onClick={() => setActiveTab('api')}
          >
            🔑 API Keys
          </button>
        </div>

        <div className="modal-body">
          {activeTab === 'tts' && (
            <div className="settings-section">
              <label className="settings-label">TTS Provider</label>
              <div className="provider-options">
                <button 
                  className={`provider-btn ${localSettings.ttsProvider === 'browser' ? 'active' : ''}`}
                  onClick={() => setLocalSettings({...localSettings, ttsProvider: 'browser'})}
                >
                  <span className="provider-icon">🌐</span>
                  <div className="provider-info">
                    <span className="provider-name">Browser</span>
                    <span className="provider-desc">Built-in voices (free)</span>
                  </div>
                </button>
                <button 
                  className={`provider-btn ${localSettings.ttsProvider === 'google' ? 'active' : ''}`}
                  onClick={() => setLocalSettings({...localSettings, ttsProvider: 'google'})}
                >
                  <span className="provider-icon">🎯</span>
                  <div className="provider-info">
                    <span className="provider-name">Google Cloud</span>
                    <span className="provider-desc">WaveNet voices (4M chars free/mo)</span>
                  </div>
                </button>
                <button 
                  className={`provider-btn ${localSettings.ttsProvider === 'elevenlabs' ? 'active' : ''}`}
                  onClick={() => setLocalSettings({...localSettings, ttsProvider: 'elevenlabs'})}
                >
                  <span className="provider-icon">✨</span>
                  <div className="provider-info">
                    <span className="provider-name">ElevenLabs</span>
                    <span className="provider-desc">Premium voices (10K chars free/mo)</span>
                  </div>
                </button>
              </div>

              {localSettings.ttsProvider === 'elevenlabs' && (
                <div className="voice-select-section">
                  <label className="settings-label">Voice</label>
                  <select 
                    className="voice-select"
                    value={localSettings.elevenlabsVoice}
                    onChange={(e) => setLocalSettings({...localSettings, elevenlabsVoice: e.target.value})}
                  >
                    <optgroup label="🎭 Character Voices">
                      <option value="ErXwobaYiN019PkySvjV">🥒 Rick Sanchez (Raspy Genius)</option>
                      <option value="ZQe5CZNOzWyzPSCn5a3c">🎸 Rockstar (Dave Grohl Style)</option>
                      <option value="pNInz6obpgDQGcFmaJgB">🤖 Adam (AI-like)</option>
                    </optgroup>
                    <optgroup label="👔 Professional">
                      <option value="21m00Tcm4TlvDq8ikWAM">Rachel (Calm & Clear)</option>
                      <option value="AZnzlk1XvdvUeBnXmlld">Domi (Strong & Confident)</option>
                      <option value="EXAVITQu4vr4xnSDxMaL">Bella (Soft & Friendly)</option>
                      <option value="MF3mGyEYCl7XYWbV9V6O">Elli (Young & Energetic)</option>
                    </optgroup>
                    <optgroup label="🎙️ Narration">
                      <option value="TxGEqnHWrfWFTfGW9XjX">Josh (Deep & Warm)</option>
                      <option value="VR6AewLTigWG4xSOukaG">Arnold (Crisp & Clear)</option>
                      <option value="pNInz6obpgDQGcFmaJgB">Adam (Deep Narrator)</option>
                      <option value="yoZ06aMxZJJ28mfd3POQ">Sam (Raspy & Cool)</option>
                    </optgroup>
                    <optgroup label="🌍 Accents">
                      <option value="jBpfuIE2acCO8z3wKNLl">Gigi (American)</option>
                      <option value="ThT5KcBeYPX3keUQqHPh">Dorothy (British)</option>
                      <option value="oWAxZDx7w5VEj9dCyTzz">Grace (Southern)</option>
                    </optgroup>
                  </select>
                  <span className="voice-hint">💡 Rick voice = raspy genius scientist vibes</span>
                </div>
              )}

              {localSettings.ttsProvider === 'google' && (
                <div className="voice-select-section">
                  <label className="settings-label">Voice</label>
                  <select 
                    className="voice-select"
                    value={localSettings.googleVoice}
                    onChange={(e) => setLocalSettings({...localSettings, googleVoice: e.target.value})}
                  >
                    <option value="en-US-Wavenet-D">US English - Male D (Wavenet)</option>
                    <option value="en-US-Wavenet-F">US English - Female F (Wavenet)</option>
                    <option value="en-US-Neural2-A">US English - Male A (Neural2)</option>
                    <option value="en-US-Neural2-C">US English - Female C (Neural2)</option>
                    <option value="en-GB-Wavenet-A">British English - Female A</option>
                    <option value="en-GB-Wavenet-B">British English - Male B</option>
                    <option value="en-AU-Wavenet-A">Australian - Female A</option>
                    <option value="en-AU-Wavenet-B">Australian - Male B</option>
                  </select>
                </div>
              )}
            </div>
          )}

          {activeTab === 'api' && (
            <div className="settings-section">
              <div className="api-key-group">
                <label className="settings-label">OpenAI API Key</label>
                <input 
                  type="password"
                  className="api-key-input"
                  placeholder="sk-..."
                  value={localSettings.openaiKey || ''}
                  onChange={(e) => setLocalSettings({...localSettings, openaiKey: e.target.value})}
                />
                <span className="api-hint">For chat with GPT models</span>
              </div>

              <div className="api-key-group">
                <label className="settings-label">ElevenLabs API Key</label>
                <input 
                  type="password"
                  className="api-key-input"
                  placeholder="xi-..."
                  value={localSettings.elevenlabsKey || ''}
                  onChange={(e) => setLocalSettings({...localSettings, elevenlabsKey: e.target.value})}
                />
                <span className="api-hint">For premium TTS voices</span>
              </div>

              <div className="api-key-group">
                <label className="settings-label">Google Cloud Credentials</label>
                <input 
                  type="text"
                  className="api-key-input"
                  placeholder="Path to service account JSON or set GOOGLE_APPLICATION_CREDENTIALS"
                  value={localSettings.googleCredentials || ''}
                  onChange={(e) => setLocalSettings({...localSettings, googleCredentials: e.target.value})}
                  disabled
                />
                <span className="api-hint">Set via environment variable on server</span>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={handleSave}>Save Settings</button>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState("You are a friendly tutor who explains sign languages and Deaf culture clearly.")
  const [useApi, setUseApi] = useState(true)
  const [isTyping, setIsTyping] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState(() => {
    const saved = localStorage.getItem('signtutor_settings')
    return saved ? JSON.parse(saved) : {
      ttsProvider: 'browser',
      elevenlabsVoice: 'ErXwobaYiN019PkySvjV', // Default to Rick!
      googleVoice: 'en-US-Wavenet-D',
      openaiKey: '',
      elevenlabsKey: '',
      googleCredentials: ''
    }
  })
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const audioRef = useRef(null)

  // Text-to-Speech function with provider support
  const speak = async (text) => {
    if (settings.ttsProvider === 'browser') {
      // Browser TTS
      if (!('speechSynthesis' in window)) {
        console.warn('Browser TTS not supported')
        return
      }
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      const voices = window.speechSynthesis.getVoices()
      const preferredVoice = voices.find(v => 
        v.name.includes('Samantha') || v.name.includes('Google') || v.lang.startsWith('en')
      )
      if (preferredVoice) utterance.voice = preferredVoice
      utterance.onstart = () => setIsSpeaking(true)
      utterance.onend = () => setIsSpeaking(false)
      utterance.onerror = () => setIsSpeaking(false)
      window.speechSynthesis.speak(utterance)
    } else {
      // Server-side TTS (Google or ElevenLabs)
      try {
        setIsSpeaking(true)
        const response = await axios.post('http://localhost:8000/tts', {
          text: text,
          provider: settings.ttsProvider,
          voice_id: settings.ttsProvider === 'elevenlabs' ? settings.elevenlabsVoice : settings.googleVoice
        }, { responseType: 'blob' })
        
        const audioUrl = URL.createObjectURL(response.data)
        if (audioRef.current) {
          audioRef.current.src = audioUrl
          audioRef.current.onended = () => {
            setIsSpeaking(false)
            URL.revokeObjectURL(audioUrl)
          }
          audioRef.current.play()
        }
      } catch (error) {
        console.error('TTS Error:', error)
        setIsSpeaking(false)
      }
    }
  }

  const stopSpeaking = () => {
    if (settings.ttsProvider === 'browser') {
      window.speechSynthesis.cancel()
    } else if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    setIsSpeaking(false)
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px'
    }
  }, [input])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      setMessages(prev => [...prev, { role: 'assistant', content: '', isLoading: true }])

      const response = await axios.post('http://localhost:8000/chat', {
        message: userMessage.content,
        system_prompt: systemPrompt,
        use_api: useApi
      })

      setMessages(prev => {
        const newMessages = [...prev]
        newMessages.pop()
        newMessages.push({ 
          role: 'assistant', 
          content: response.data.response,
          source: response.data.source,
          isNew: true
        })
        return newMessages
      })
      setIsTyping(true)
      
      // Start TTS immediately when response arrives (plays while text streams)
      if (ttsEnabled) {
        speak(response.data.response)
      }
      
      const wordCount = response.data.response.split(/\s+/).length
      const streamDuration = wordCount * 40 + 800
      setTimeout(() => {
        setMessages(prev => prev.map((msg, idx) => 
          idx === prev.length - 1 ? { ...msg, isNew: false } : msg
        ))
        setIsTyping(false)
      }, streamDuration)

    } catch (error) {
      console.error("Error:", error)
      const errorMsg = error.response?.data?.detail || 'Unable to connect.'
      setMessages(prev => {
        const newMessages = [...prev]
        newMessages.pop()
        newMessages.push({ role: 'assistant', content: errorMsg, isError: true })
        return newMessages
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(e)
    }
  }

  const clearChat = () => setMessages([])

  return (
    <div className="app-wrapper">
      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} />
      
      {/* Settings Modal */}
      <SettingsModal 
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onUpdateSettings={setSettings}
      />

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">🤟</span>
            <span className="logo-text">SignTutor</span>
          </div>
        </div>

        <button className="new-chat-btn" onClick={clearChat}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New Conversation
        </button>

        <div className="sidebar-content">
          <div className="sidebar-section">
            <label className="section-label">Model</label>
            <div className="model-selector">
              <button className={`model-option ${useApi ? 'active' : ''}`} onClick={() => setUseApi(true)}>
                <span className="model-icon">✨</span>
                <div className="model-info">
                  <span className="model-name">Cloud AI</span>
                  <span className="model-desc">GPT-3.5 Turbo</span>
                </div>
              </button>
              <button className={`model-option ${!useApi ? 'active' : ''}`} onClick={() => setUseApi(false)}>
                <span className="model-icon">🖥️</span>
                <div className="model-info">
                  <span className="model-name">Local</span>
                  <span className="model-desc">Qwen3-4B</span>
                </div>
              </button>
            </div>
          </div>

          <div className="sidebar-section">
            <label className="section-label">Persona</label>
            <textarea
              className="persona-input"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={3}
              placeholder="Describe how the AI should behave..."
            />
          </div>

          <div className="sidebar-section">
            <label className="section-label">Text-to-Speech</label>
            <button 
              className={`tts-toggle ${ttsEnabled ? 'active' : ''}`}
              onClick={() => {
                if (ttsEnabled && isSpeaking) stopSpeaking()
                setTtsEnabled(!ttsEnabled)
              }}
            >
              <span className="tts-icon">{ttsEnabled ? '🔊' : '🔇'}</span>
              <span className="tts-label">
                {ttsEnabled ? `On (${settings.ttsProvider})` : 'Disabled'}
              </span>
            </button>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="footer-row">
            <div className="status-badge">
              <span className={`status-dot ${useApi ? 'cloud' : 'local'}`}></span>
              {useApi ? 'Cloud' : 'Local'}
            </div>
            <button className="settings-btn" onClick={() => setShowSettings(true)} title="Settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-content">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-screen">
              <div className="welcome-card">
                <div className="welcome-icon-wrapper">
                  <span className="welcome-icon">🤟</span>
                </div>
                <h1>Welcome to SignTutor</h1>
                <p>Your personal guide to sign languages and Deaf culture. Ask anything!</p>
                
                <div className="prompt-grid">
                  <button className="prompt-card" onClick={() => setInput("What's the difference between ASL and BSL?")}>
                    <span className="prompt-icon">🌍</span>
                    <span>Compare ASL and BSL</span>
                  </button>
                  <button className="prompt-card" onClick={() => setInput("Teach me about Deaf culture and community")}>
                    <span className="prompt-icon">🤝</span>
                    <span>Deaf Culture & Community</span>
                  </button>
                  <button className="prompt-card" onClick={() => setInput("How do I sign common greetings?")}>
                    <span className="prompt-icon">👋</span>
                    <span>Learn Basic Greetings</span>
                  </button>
                  <button className="prompt-card" onClick={() => setInput("What resources can help me learn sign language?")}>
                    <span className="prompt-icon">📚</span>
                    <span>Learning Resources</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message-row ${msg.role}`}>
              <div className="message-wrapper">
                <div className="avatar-container">
                  {msg.role === 'user' ? (
                    <div className="avatar user-avatar">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                      </svg>
                    </div>
                  ) : (
                    <div className="avatar assistant-avatar">
                      <span>🤟</span>
                    </div>
                  )}
                </div>
                <div className="message-content">
                  <div className="message-header">
                    <span className="sender-name">{msg.role === 'user' ? 'You' : 'SignTutor'}</span>
                    {msg.source && <span className="source-tag">{msg.source}</span>}
                    {msg.role === 'assistant' && !msg.isLoading && !msg.isNew && (
                      <button 
                        className={`speak-btn ${isSpeaking ? 'speaking' : ''}`}
                        onClick={() => isSpeaking ? stopSpeaking() : speak(msg.content)}
                        title={isSpeaking ? 'Stop' : 'Read aloud'}
                      >
                        {isSpeaking ? '⏹️' : '🔊'}
                      </button>
                    )}
                  </div>
                  {msg.isLoading ? (
                    <div className="typing-indicator">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  ) : (
                    <div className={`message-text ${msg.isError ? 'error' : ''}`}>
                      {msg.role === 'assistant' && msg.isNew ? (
                        <StreamingText text={msg.content} speed={40} />
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-wrapper">
          <div className="input-container">
            <form className="input-form" onSubmit={sendMessage}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me anything about sign language..."
                disabled={isLoading}
                rows={1}
              />
              <button type="submit" disabled={isLoading || !input.trim()} className="send-btn">
                {isLoading ? (
                  <div className="loading-spinner"></div>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 2L11 13"></path>
                    <path d="M22 2L15 22L11 13L2 9L22 2Z"></path>
                  </svg>
                )}
              </button>
            </form>
          </div>
          <p className="footer-text">SignTutor may produce inaccurate information. Please verify important details.</p>
        </div>
      </main>
    </div>
  )
}

export default App
