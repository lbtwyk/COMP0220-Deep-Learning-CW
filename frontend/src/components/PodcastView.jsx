import React, { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import axios from 'axios'

/**
 * PodcastView - Agentic podcast with TTS support
 */

export default function PodcastView({ onBack, settings, onOpenSettings }) {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [podcastState, setPodcastState] = useState('idle')
  const [currentTopic, setCurrentTopic] = useState('')
  const [topicInput, setTopicInput] = useState('')
  const [personality, setPersonality] = useState('professional')
  const [messages, setMessages] = useState([])
  const [interruptInput, setInterruptInput] = useState('')
  const [speakingAgent, setSpeakingAgent] = useState(null)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(true)
  const [isSwitchingPersonality, setIsSwitchingPersonality] = useState(false)
  const [modelType, setModelType] = useState('api') // 'api', 'local', or 'lstm'

  const audioRef = useRef(null)
  const wsRef = useRef(null)
  const messagesEndRef = useRef(null)
  const speechQueueRef = useRef([])
  const isProcessingRef = useRef(false)
  const handleMessageRef = useRef(null)

  // TopMediai voice IDs (from TopMediai website)
  const TOPMEDIAI_RICK_VOICE = '67ad973f-5d4b-11ee-a861-00163e2ac61b'  // Rick Sanchez
  const TOPMEDIAI_MORTY_VOICE = '67ada016-5d4b-11ee-a861-00163e2ac61b' // Morty Smith

  // Agent voice mapping
  const agentVoices = {
    fun: {
      rick: TOPMEDIAI_RICK_VOICE,      // TopMediai Rick
      morty: TOPMEDIAI_MORTY_VOICE,    // TopMediai Morty
      summer: 'MF3mGyEYCl7XYWbV9V6O',  // Elli (ElevenLabs)
    },
    professional: {
      rick: 'TxGEqnHWrfWFTfGW9XjX',    // Josh (ElevenLabs)
      morty: '21m00Tcm4TlvDq8ikWAM',   // Rachel (ElevenLabs)
      summer: 'EXAVITQu4vr4xnSDxMaL',  // Bella (ElevenLabs)
    }
  }

  // Google fallback voices
  const googleFallbackVoices = {
    rick: 'en-US-Wavenet-D',    // Male voice
    morty: 'en-US-Wavenet-F',   // Female voice (higher pitch)
    summer: 'en-US-Wavenet-A',  // Neutral
  }

  // TTS function - speaks text for an agent
  const speak = useCallback(async (text, agent) => {
    if (!ttsEnabled) {
      console.log(`🔇 TTS disabled, skipping speech for ${agent}`)
      return
    }

    const provider = settings?.ttsProvider || 'browser'

    // Rick and Morty ALWAYS use server TTS (TopMediai/Google), never browser
    if (agent === 'rick' || agent === 'morty') {
      console.log(`🎙️ ${agent} requires server TTS, forcing server mode`)
      // Force server TTS for Rick/Morty
    } else if (provider === 'browser') {
      // Browser TTS only for Summer
      if (!('speechSynthesis' in window)) return

      return new Promise((resolve) => {
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.rate = agent === 'summer' ? 1.1 : 1.0
        utterance.pitch = agent === 'summer' ? 1.1 : 1.0

        const voices = window.speechSynthesis.getVoices()
        const preferredVoice = voices.find(v => v.lang.startsWith('en'))
        if (preferredVoice) utterance.voice = preferredVoice

        utterance.onstart = () => {
          setIsSpeaking(true)
          setSpeakingAgent(agent)
        }
        utterance.onend = () => {
          setIsSpeaking(false)
          setSpeakingAgent(null)
          resolve()
        }
        utterance.onerror = () => {
          setIsSpeaking(false)
          setSpeakingAgent(null)
          resolve()
        }

        window.speechSynthesis.speak(utterance)
      })
    }

    // Server TTS (TopMediai/ElevenLabs/Google) - ALWAYS used for Rick/Morty
    // (Rick/Morty skip browser TTS, Summer can use browser if selected)
    try {
      setIsSpeaking(true)
      setSpeakingAgent(agent)

      // Determine provider and voice for this agent
      let ttsProvider = provider
      let voiceId = null
      let fallbackVoice = null

      // Rick and Morty use Google TTS directly (TopMediai is too slow/unreliable)
      if (agent === 'rick' || agent === 'morty') {
        ttsProvider = 'google'  // Use Google directly - TopMediai is too slow
        // Use distinct Google voices for Rick and Morty
        voiceId = agent === 'rick' ? 'en-US-Wavenet-D' : 'en-US-Wavenet-F'  // D = deeper (Rick), F = higher (Morty)
        fallbackVoice = voiceId
        console.log(`🎙️ Using Google Cloud TTS for ${agent} (voice: ${voiceId})`)
      } else {
        // Summer uses ElevenLabs or Google (or browser if selected)
        if (provider === 'elevenlabs') {
          ttsProvider = 'elevenlabs'
          voiceId = agentVoices[personality]?.[agent] || settings?.elevenlabsVoice
        } else if (provider === 'google') {
          ttsProvider = 'google'
          voiceId = settings?.googleVoice || googleFallbackVoices[agent]
        } else {
          // Browser TTS for Summer (already handled above)
          return Promise.resolve()
        }
      }

      console.log(`📡 TTS Request: provider=${ttsProvider}, agent=${agent}, voiceId=${voiceId}, textLength=${text.length}`)

      let response
      try {
        response = await axios.post('http://localhost:8000/tts', {
          text: text,
          provider: ttsProvider,
          voice_id: voiceId,
          fallback_to_google: false, // Not needed since we're using Google directly
          google_fallback_voice: fallbackVoice,
          speed: 1.4, // Faster reading speed (1.4x normal speed)
        }, {
          responseType: 'blob',
          timeout: 60000 // 60 second timeout for TopMediai
        })

        console.log(`✅ TTS Response received: ${response.data.size} bytes, type: ${response.data.type || 'unknown'}`)
      } catch (ttsError) {
        console.error(`❌ TTS request failed:`, ttsError)
        throw ttsError
      }

      return new Promise((resolve) => {
        const audioUrl = URL.createObjectURL(response.data)
        console.log(`🔊 Created audio URL: ${audioUrl.substring(0, 50)}...`)

        if (audioRef.current) {
          audioRef.current.src = audioUrl
          audioRef.current.onloadeddata = () => {
            console.log(`✅ Audio loaded, duration: ${audioRef.current.duration}s`)
          }
          audioRef.current.onplay = () => {
            console.log(`▶️ Audio playing for ${agent}`)
          }
          audioRef.current.onended = () => {
            console.log(`⏹️ Audio ended for ${agent}`)
            setIsSpeaking(false)
            setSpeakingAgent(null)
            URL.revokeObjectURL(audioUrl)
            resolve()
          }
          audioRef.current.onerror = (e) => {
            console.error(`❌ Audio playback error for ${agent}:`, e, audioRef.current.error)
            setIsSpeaking(false)
            setSpeakingAgent(null)
            URL.revokeObjectURL(audioUrl)
            resolve()
          }

          const playPromise = audioRef.current.play()
          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                console.log(`✅ Audio play() succeeded for ${agent}`)
              })
              .catch((playError) => {
                console.error(`❌ Audio play() failed for ${agent}:`, playError)
                resolve()
              })
          }
        } else {
          console.error(`❌ audioRef.current is null!`)
          resolve()
        }
      })
    } catch (error) {
      console.error('TTS Error:', error)
      console.error('Error details:', error.response?.data || error.message)

      // If Google fails for Rick/Morty, try browser TTS as last resort
      if (agent === 'rick' || agent === 'morty') {
        console.log('Google TTS failed, trying browser TTS as last resort...')
        try {
          if ('speechSynthesis' in window) {
            return new Promise((resolve) => {
              const utterance = new SpeechSynthesisUtterance(text)
              utterance.rate = agent === 'rick' ? 1.0 : 1.1
              utterance.pitch = agent === 'morty' ? 1.2 : 0.9

              const voices = window.speechSynthesis.getVoices()
              const preferredVoice = voices.find(v => v.lang.startsWith('en'))
              if (preferredVoice) utterance.voice = preferredVoice

              utterance.onstart = () => {
                setIsSpeaking(true)
                setSpeakingAgent(agent)
              }
              utterance.onend = () => {
                setIsSpeaking(false)
                setSpeakingAgent(null)
                resolve()
              }
              utterance.onerror = () => {
                setIsSpeaking(false)
                setSpeakingAgent(null)
                resolve()
              }

              window.speechSynthesis.speak(utterance)
            })
          }
        } catch (browserError) {
          console.error('Browser TTS fallback also failed:', browserError)
        }
      }

      setIsSpeaking(false)
      setSpeakingAgent(null)
    }
  }, [ttsEnabled, settings, personality])

  // Process speech queue sequentially
  const processSpeechQueue = useCallback(async () => {
    if (isProcessingRef.current || speechQueueRef.current.length === 0) return

    isProcessingRef.current = true
    while (speechQueueRef.current.length > 0) {
      const { text, agent } = speechQueueRef.current.shift()
      await speak(text, agent)
      await new Promise(r => setTimeout(r, 300)) // Small pause between speeches
    }
    isProcessingRef.current = false
  }, [speak])

  // Queue speech
  const queueSpeech = useCallback((text, agent) => {
    speechQueueRef.current.push({ text, agent })
    processSpeechQueue()
  }, [processSpeechQueue])

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle WebSocket messages (defined before connect to avoid circular dependency)
  const handleMessage = useCallback((msg) => {
    console.log('📨 Received WebSocket message:', msg.type, msg)

    switch (msg.type) {
      case 'agents_info':
        // Update personality and re-render messages with new names
        const newPersonality = msg.personality || 'professional' // Default to professional if not specified
        setPersonality(newPersonality) // Always update to match backend
        setIsSwitchingPersonality(false) // Switch complete
        // Force re-render by creating new message objects
        setMessages(prev => prev.map(m => ({
          ...m,
          _personality: newPersonality // Add marker to force re-render
        })))
        break
      case 'speech':
        console.log(`💬 Speech from ${msg.agent}:`, msg.text.substring(0, 50) + '...')
        setMessages(prev => {
          const newMessages = [...prev, {
            role: msg.agent,
            content: msg.text,
            model: msg.model
          }]
          console.log(`📝 Total messages: ${newMessages.length}`)
          return newMessages
        })
        // Queue TTS
        queueSpeech(msg.text, msg.agent)
        break
      case 'state':
        console.log(`🔄 State change: ${msg.state}`, msg.topic ? `(topic: ${msg.topic})` : '')
        setPodcastState(msg.state)
        if (msg.topic) setCurrentTopic(msg.topic)
        break
      case 'request_topic':
        console.log('❓ Requesting topic from user')
        setPodcastState('topic_input')
        break
      case 'ended':
        console.log('🏁 Podcast ended')
        stopSpeaking() // Stop any audio immediately
        setPodcastState('idle')
        break
      default:
        console.log('⚠️ Unknown message type:', msg.type)
    }
  }, [personality, queueSpeech])

  // Update ref when handleMessage changes
  useEffect(() => {
    handleMessageRef.current = handleMessage
  }, [handleMessage])

  // Connect WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setIsConnecting(true)
    const ws = new WebSocket('ws://localhost:8000/ws/podcast')
    const currentPersonality = personality || 'professional'

    ws.onopen = () => {
      setIsConnected(true)
      setIsConnecting(false)
      // Send current personality to backend on connect
      ws.send(JSON.stringify({ type: 'set_personality', personality: currentPersonality }))
      // Sync model preference
      ws.send(JSON.stringify({ type: 'set_model', model_type: modelType }))
    }

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      // Use ref to get latest handleMessage (avoids circular dependency)
      if (handleMessageRef.current) {
        handleMessageRef.current(msg)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      setIsConnecting(false)
      setPodcastState('idle')
    }

    ws.onerror = () => setIsConnecting(false)
    wsRef.current = ws
  }, [personality, modelType])

  const send = (type, data = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, ...data }))
    }
  }

  const startPodcast = () => {
    if (!topicInput.trim()) return
    send('start', { topic: topicInput.trim() })
    setTopicInput('')
  }

  const sendInterrupt = () => {
    if (!interruptInput.trim()) return
    stopSpeaking() // Immediately stop current speech and clear queue
    send('interrupt', { message: interruptInput.trim() })
    setMessages(prev => [...prev, { role: 'user', content: interruptInput.trim() }])
    setInterruptInput('')
  }

  const stopSpeaking = () => {
    window.speechSynthesis?.cancel()
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0

      // Force resolve the current promise if there's an active handler
      // This is crucial to unblock processSpeechQueue
      if (typeof audioRef.current.onended === 'function') {
        audioRef.current.onended()
      }
    }
    speechQueueRef.current = []
    setIsSpeaking(false)
    setSpeakingAgent(null)

    // Safety: ensure processing flag is cleared if we forced a stop
    // (Though the onended callback should handle this via the loop exit)
  }

  useEffect(() => {
    return () => {
      wsRef.current?.close()
      stopSpeaking()
    }
  }, [])

  const getAgentInfo = (role) => {
    if (personality === 'fun') {
      return {
        rick: { name: 'Rick', emoji: '🥒', color: 'rick' },
        morty: { name: 'Morty', emoji: '😰', color: 'morty' },
        summer: { name: 'Summer', emoji: '🎯', color: 'summer' },
        user: { name: 'You', emoji: '👤', color: 'user' }
      }[role] || { name: role, emoji: '🤖', color: 'user' }
    }
    return {
      rick: { name: 'Dave', emoji: '🎸', color: 'rick' },
      morty: { name: 'Taylor', emoji: '🥁', color: 'morty' },
      summer: { name: 'Pat', emoji: '🎸', color: 'summer' },
      user: { name: 'You', emoji: '👤', color: 'user' }
    }[role] || { name: role, emoji: '🤖', color: 'user' }
  }

  return (
    <div className="app-wrapper">
      <audio ref={audioRef} />

      {/* Sidebar - same as chat */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">🤟</span>
            <span className="logo-text">SignTutor</span>
          </div>
        </div>

        <button className="new-chat-btn" onClick={onBack}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Back to Chat
        </button>

        <div className="sidebar-content">
          <div className="sidebar-section">
            <label className="section-label">Personality</label>
            <div className="model-selector">
              <button
                className={`model-option ${personality === 'fun' ? 'active' : ''}`}
                disabled={isSwitchingPersonality}
                onClick={() => {
                  if (personality !== 'fun') {
                    if (isConnected) {
                      setIsSwitchingPersonality(true)
                      send('set_personality', { personality: 'fun' })
                    } else {
                      // Allow switching even when not connected (will apply on connect)
                      setPersonality('fun')
                    }
                  }
                }}
              >
                <span className="model-icon">🥒</span>
                <div className="model-info">
                  <span className="model-name">Fun Mode</span>
                  <span className="model-desc">Rick & Morty</span>
                </div>
              </button>
              <button
                className={`model-option ${personality === 'professional' ? 'active' : ''}`}
                disabled={isSwitchingPersonality}
                onClick={() => {
                  if (personality !== 'professional') {
                    if (isConnected) {
                      setIsSwitchingPersonality(true)
                      send('set_personality', { personality: 'professional' })
                    } else {
                      // Allow switching even when not connected (will apply on connect)
                      setPersonality('professional')
                    }
                  }
                }}
              >
                <span className="model-icon">👔</span>
                <div className="model-info">
                  <span className="model-name">Professional</span>
                  <span className="model-desc">Educational</span>
                </div>
              </button>
            </div>
            {isSwitchingPersonality && (
              <div style={{ fontSize: '0.8em', color: 'var(--text-secondary)', marginTop: '8px', textAlign: 'center' }}>
                Switching personality...
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <label className="section-label">AI Model</label>
            <div className="model-selector">
              <button
                className={`model-option ${modelType === 'api' ? 'active' : ''}`}
                onClick={() => {
                  if (modelType !== 'api') {
                    setModelType('api')
                    if (isConnected) {
                      send('set_model', { model_type: 'api' })
                    }
                  }
                }}
              >
                <span className="model-icon">☁️</span>
                <div className="model-info">
                  <span className="model-name">Cloud AI</span>
                  <span className="model-desc">GPT-4o-mini</span>
                </div>
              </button>
              <button
                className={`model-option ${modelType === 'local' ? 'active' : ''}`}
                onClick={() => {
                  if (modelType !== 'local') {
                    setModelType('local')
                    if (isConnected) {
                      send('set_model', { model_type: 'local' })
                    }
                  }
                }}
              >
                <span className="model-icon">🖥️</span>
                <div className="model-info">
                  <span className="model-name">Local</span>
                  <span className="model-desc">Qwen3-4B</span>
                </div>
              </button>
              <button
                className={`model-option ${modelType === 'lstm' ? 'active' : ''}`}
                onClick={() => {
                  if (modelType !== 'lstm') {
                    setModelType('lstm')
                    if (isConnected) {
                      send('set_model', { model_type: 'lstm' })
                    }
                  }
                }}
              >
                <span className="model-icon">🤪</span>
                <div className="model-info">
                  <span className="model-name">LSTM</span>
                  <span className="model-desc">Dummy Model</span>
                </div>
              </button>
            </div>
          </div>

          <div className="sidebar-section">
            <label className="section-label">Text-to-Speech</label>
            <button
              className={`tts-toggle ${ttsEnabled ? 'active' : ''}`}
              onClick={() => {
                if (ttsEnabled) stopSpeaking()
                setTtsEnabled(!ttsEnabled)
              }}
            >
              <span className="tts-icon">{ttsEnabled ? '🔊' : '🔇'}</span>
              <span className="tts-label">
                {ttsEnabled ? `On (${settings?.ttsProvider || 'browser'})` : 'Disabled'}
              </span>
            </button>
          </div>

          {currentTopic && (
            <div className="sidebar-section">
              <label className="section-label">Current Topic</label>
              <div className="current-topic-display">{currentTopic}</div>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="footer-row">
            <div className="status-badge">
              <span className={`status-dot ${isConnected ? 'cloud' : ''}`}></span>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            <button className="settings-btn" onClick={onOpenSettings} title="Settings">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Podcast Mode Banner */}
        <div className="podcast-banner">
          <span className="podcast-banner-icon">🎙️</span>
          <span className="podcast-banner-text">
            <strong>Podcast Mode</strong> — {personality === 'fun' ? 'Rick & Morty' : 'Dave & Taylor'} discussing sign language
          </span>
          {(podcastState === 'discussing' || isSpeaking) && (
            <button
              className="podcast-banner-stop"
              onClick={() => {
                // Stop any current audio immediately
                stopSpeaking()
                // Tell backend to end the session so the loop stops
                send('end')
                // Reset state immediately
                setPodcastState('idle')
              }}
            >
              ⏹️ Stop
            </button>
          )}
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-screen">
              <div className="welcome-card">
                <div className="welcome-icon-wrapper">
                  <span className="welcome-icon">🎙️</span>
                </div>
                <h1>{personality === 'fun' ? 'Rick & Morty Podcast' : 'SignTutor Podcast with Dave & Taylor'}</h1>
                <p>
                  {personality === 'fun'
                    ? "Wubba lubba dub dub! Let's learn about sign language!"
                    : "Learn about sign language and Deaf culture with our hosts"}
                </p>

                {!isConnected ? (
                  <button
                    className="prompt-card"
                    onClick={connect}
                    disabled={isConnecting}
                    style={{ width: '100%', justifyContent: 'center' }}
                  >
                    <span className="prompt-icon">🎙️</span>
                    <span>{isConnecting ? 'Connecting...' : 'Start Podcast'}</span>
                  </button>
                ) : (
                  <div className="prompt-grid">
                    <button className="prompt-card" onClick={() => setTopicInput("What is Deaf culture?")}>
                      <span className="prompt-icon">🤝</span>
                      <span>Deaf Culture</span>
                    </button>
                    <button className="prompt-card" onClick={() => setTopicInput("Teach me ASL basics")}>
                      <span className="prompt-icon">👋</span>
                      <span>ASL Basics</span>
                    </button>
                    <button className="prompt-card" onClick={() => setTopicInput("How does ASL grammar work?")}>
                      <span className="prompt-icon">📚</span>
                      <span>ASL Grammar</span>
                    </button>
                    <button className="prompt-card" onClick={() => setTopicInput("What is a CODA?")}>
                      <span className="prompt-icon">👨‍👩‍👧</span>
                      <span>CODA Families</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {messages.map((msg, index) => {
            const info = getAgentInfo(msg.role)
            // Strip agent tags like [RICK]: or [MORTY]: from content
            let displayContent = msg.content
            if (displayContent && displayContent.match(/^\[(RICK|MORTY|SUMMER|DAVE|TAYLOR|PAT|ALEX|MORGAN|SAM)\]:\s*/i)) {
              displayContent = displayContent.replace(/^\[(RICK|MORTY|SUMMER|DAVE|TAYLOR|PAT|ALEX|MORGAN|SAM)\]:\s*/i, '')
            }

            return (
              <div key={index} className={`message-row ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                <div className="message-wrapper">
                  <div className="avatar-container">
                    <div className={`avatar podcast-avatar-${info.color} ${speakingAgent === msg.role ? 'speaking' : ''}`}>
                      <span>{info.emoji}</span>
                    </div>
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="sender-name">{info.name}</span>
                      {msg.model && (
                        <span className="model-badge" title={`Generated by ${msg.model === 'local' ? 'Local Qwen3' : msg.model === 'lstm' ? 'LSTM Model' : 'Cloud OpenAI'}`}>
                          {msg.model === 'local' ? '🖥️ Local' : msg.model === 'lstm' ? '🤪 LSTM' : '☁️ Cloud'}
                        </span>
                      )}
                      {speakingAgent === msg.role && <span className="speaking-indicator">🔊</span>}
                    </div>
                    <div className={`message-text podcast-bubble-${info.color}`}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-wrapper">
          <div className="input-container">
            {!isConnected ? (
              <button
                className="send-btn"
                onClick={connect}
                disabled={isConnecting}
                style={{ width: '100%', borderRadius: '12px', padding: '16px' }}
              >
                {isConnecting ? 'Connecting...' : '🎙️ Connect to Podcast'}
              </button>
            ) : podcastState === 'topic_input' || podcastState === 'idle' ? (
              <form className="input-form" onSubmit={(e) => { e.preventDefault(); startPodcast(); }}>
                <textarea
                  value={topicInput}
                  onChange={(e) => setTopicInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startPodcast(); } }}
                  placeholder="Enter a topic to discuss..."
                  rows={1}
                />
                <button type="submit" disabled={!topicInput.trim()} className="send-btn">
                  🎙️
                </button>
              </form>
            ) : (
              <form className="input-form" onSubmit={(e) => { e.preventDefault(); sendInterrupt(); }}>
                <textarea
                  value={interruptInput}
                  onChange={(e) => setInterruptInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendInterrupt(); } }}
                  placeholder="Ask a question to interrupt..."
                  rows={1}
                />
                <button type="submit" disabled={!interruptInput.trim()} className="send-btn">
                  ✋
                </button>
              </form>
            )}
          </div>
          <p className="footer-text">
            {personality === 'fun'
              ? "🥒 Rick knows everything. Morty asks the questions. You can interrupt anytime!"
              : "Ask questions anytime to join the conversation"}
          </p>
        </div>
      </main>
    </div>
  )
}

