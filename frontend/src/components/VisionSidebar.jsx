/**
 * VisionSidebar Component
 * Real-time ASL letter recognition with webcam, letter buffer, and word suggestions
 */
import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useVisionWebSocket } from '../hooks/useVisionWebSocket'
import './VisionSidebar.css'
import './SettingsControls.css'

export const VisionSidebar = ({ onInterrupt, isOpen, onToggle }) => {
    const [buffer, setBuffer] = useState([])
    const [currentLetter, setCurrentLetter] = useState(null)
    const [confidence, setConfidence] = useState(0)
    const [suggestions, setSuggestions] = useState([])
    const [webcamActive, setWebcamActive] = useState(false)
    const [error, setError] = useState(null)
    const [fps, setFps] = useState(5) // Default 5 FPS

    const videoRef = useRef()
    const canvasRef = useRef()
    const streamRef = useRef(null)
    const frameIntervalRef = useRef(null)

    // Calculate interval from FPS
    const frameInterval = 1000 / fps // ms per frame

    // WebSocket connection
    const { sendFrame, clearBuffer, deleteLast, sendInterrupt, isConnected } = useVisionWebSocket({
        onLetterUpdate: (data) => {
            if (data.type === 'letter_update') {
                setCurrentLetter(data.letter)
                setConfidence(data.confidence)
                setBuffer(data.buffer || [])
                setSuggestions(data.suggestions || [])
            } else if (data.type === 'no_hand') {
                setCurrentLetter(null)
                setConfidence(0)
                setBuffer(data.buffer || [])
                setSuggestions(data.suggestions || [])
            } else if (data.type === 'buffer_cleared' || data.type === 'letter_deleted') {
                setBuffer(data.buffer || [])
                setSuggestions(data.suggestions || [])
            } else if (data.type === 'interrupt_ready') {
                // Send to podcast
                if (onInterrupt) {
                    onInterrupt(data.message)
                }
                setBuffer([])
                setSuggestions([])
            }
        },
        onError: (errorMsg) => {
            setError(errorMsg)
            setTimeout(() => setError(null), 5000)
        }
    })

    // Start webcam
    const startWebcam = useCallback(async () => {
        console.log('🎥 Starting webcam...')
        try {
            console.log('📹 Requesting camera permissions...')
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }  // Match cv2.VideoCapture default
            })

            console.log('✅ Camera stream obtained:', stream)
            console.log('Video tracks:', stream.getVideoTracks())

            if (videoRef.current) {
                console.log('📺 Setting srcObject on video element')
                videoRef.current.srcObject = stream
                streamRef.current = stream

                // Wait for video to be ready and explicitly play
                videoRef.current.onloadedmetadata = () => {
                    console.log('📼 Video metadata loaded, playing...')
                    videoRef.current.play()
                        .then(() => {
                            console.log('▶️ Video playing successfully!')
                            setWebcamActive(true)
                            setError(null)
                        })
                        .catch(err => {
                            console.error('❌ Play error:', err)
                            setError('Could not start video playback')
                        })
                }
            } else {
                console.error('❌ videoRef.current is null')
            }
        } catch (err) {
            console.error('❌ Webcam error:', err)
            console.error('Error name:', err.name)
            console.error('Error message:', err.message)
            setError(`Could not access webcam: ${err.message}`)
        }
    }, [])

    // Stop webcam
    const stopWebcam = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop())
            streamRef.current = null
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
        setWebcamActive(false)
    }, [])

    // Capture and send frames
    const captureFrame = useCallback(() => {
        if (!videoRef.current || !canvasRef.current || !isConnected) {
            console.log('⏭️ Skipping frame:', {
                hasVideo: !!videoRef.current,
                hasCanvas: !!canvasRef.current,
                isConnected
            })
            return
        }

        const video = videoRef.current
        const canvas = canvasRef.current

        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            const ctx = canvas.getContext('2d')
            ctx.drawImage(video, 0, 0, 640, 480)

            // Convert to base64
            const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1]
            console.log('📸 Sending frame, size:', base64.length, 'chars')
            sendFrame(base64)
        } else {
            console.log('⏸️ Video not ready, state:', video.readyState)
        }
    }, [sendFrame, isConnected])

    // Start/stop frame capture
    useEffect(() => {
        if (webcamActive && isConnected) {
            // Send frames at selected FPS
            console.log(`📸 Starting frame capture at ${fps} FPS (${frameInterval}ms interval)`)
            frameIntervalRef.current = setInterval(captureFrame, frameInterval)
        } else {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current)
                frameIntervalRef.current = null
            }
        }

        return () => {
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current)
            }
        }
    }, [webcamActive, isConnected, captureFrame, fps, frameInterval])

    // Cleanup on unmount and auto-start on open
    useEffect(() => {
        const startCam = async () => {
            if (!webcamActive) {
                startWebcam()
            }
        }

        if (isOpen) {
            startCam()
        }

        return () => {
            // Cleanup on close or unmount
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop())
                streamRef.current = null
            }
            if (videoRef.current) {
                videoRef.current.srcObject = null
            }
            setWebcamActive(false)
        }
    }, [isOpen]) // eslint-disable-line react-hooks/exhaustive-deps

    const handleSendQuestion = () => {
        const word = buffer.join('')
        if (word.trim()) {
            console.log('📤 Sending ASL word to podcast:', word)
            onInterrupt(`What does ${word.toUpperCase()} mean in ASL?`)
            clearBuffer()
        }
    }

    // Render toggle button for embedding in parent component
    if (!isOpen) {
        return (
            <button className="vision-toggle-btn" onClick={onToggle} title="Open Sign Language Input">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                </svg>
            </button>
        )
    }

    return (
        <div className="vision-sidebar">
            <div className="vision-header">
                <h3>📹 Sign Language Input</h3>
                <button className="vision-close-btn" onClick={onToggle}>×</button>
            </div>

            {/* Webcam Preview */}
            <div className="webcam-container">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    width="224"
                    height="224"
                    style={{ display: webcamActive ? 'block' : 'none' }}
                />
                <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />

                {!webcamActive && (
                    <div className="webcam-placeholder">
                        <button onClick={startWebcam} className="start-webcam-btn">
                            Start Camera
                        </button>
                    </div>
                )}

                {/* Connection Status */}
                <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                    <span className="status-dot" />
                    {isConnected ? 'Connected' : 'Connecting...'}
                </div>
            </div>

            {/* FPS Selector */}
            <div className="fps-selector">
                <label className="fps-label">Frame Rate</label>
                <div className="segmented-control">
                    {[3, 5, 10].map(fpsOption => (
                        <button
                            key={fpsOption}
                            className={`segment ${fps === fpsOption ? 'active' : ''}`}
                            onClick={() => setFps(fpsOption)}
                        >
                            {fpsOption} FPS
                        </button>
                    ))}
                </div>
            </div>

            {/* Current Letter */}
            <div className="current-letter">
                <div className="letter-display">
                    {currentLetter ? currentLetter.toUpperCase() : '—'}
                </div>
                <div className="confidence-container">
                    <div className="confidence-bar">
                        <div
                            className="confidence-fill"
                            style={{ width: `${confidence * 100}%` }}
                        />
                    </div>
                    <span className="confidence-text">
                        {currentLetter ? `${(confidence * 100).toFixed(0)}%` : 'No hand'}
                    </span>
                </div>
            </div>

            {/* Letter Buffer */}
            <div className="letter-buffer">
                <div className="buffer-header">
                    <span>Buffer:</span>
                    <button
                        className="delete-btn"
                        onClick={deleteLast}
                        disabled={buffer.length === 0}
                        title="Delete last letter"
                    >
                        ⌫
                    </button>
                </div>
                <div className="buffer-letters">
                    {buffer.length > 0 ? (
                        buffer.map((letter, i) => (
                            <span key={i} className="buffer-letter">{letter.toUpperCase()}</span>
                        ))
                    ) : (
                        <span className="buffer-empty">Empty</span>
                    )}
                </div>
            </div>

            {/* Suggestions */}
            {suggestions.length > 0 && (
                <div className="suggestions">
                    <div className="suggestions-header">💡 Suggestions:</div>
                    <div className="suggestions-list">
                        {suggestions.map((word, i) => (
                            <button
                                key={i}
                                className="suggestion-btn"
                                onClick={() => {
                                    // Could implement: select suggestion
                                    console.log('Selected:', word)
                                }}
                            >
                                {word}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Error Message */}
            {error && (
                <div className="error-message">
                    ⚠️ {error}
                </div>
            )}

            {/* Controls */}
            <div className="vision-controls">
                <button
                    className="control-btn secondary"
                    onClick={clearBuffer}
                >
                    Clear
                </button>
                <button
                    className="control-btn primary"
                    onClick={handleSendQuestion}
                    disabled={buffer.length === 0}
                >
                    Send Question
                </button>
            </div>

            {/* Help Text */}
            <div className="vision-help">
                <small>
                    Sign ASL letters (A-Y) to form words. Recognized letters appear in the buffer.
                </small>
            </div>
        </div>
    )
}
