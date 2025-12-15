/**
 * RecordingIndicator - Apple-style recording badge
 */
import React from 'react'
import './RecordingIndicator.css'

export const RecordingIndicator = ({ isRecording, onToggle, showLabel = true }) => {
    return (
        <div className={`recording-indicator ${isRecording ? 'active' : ''}`}>
            <button
                className="rec-button"
                onClick={onToggle}
                title={isRecording ? 'Stop Recording' : 'Start Recording'}
            >
                <div className="rec-icon">
                    <div className="rec-dot" />
                </div>
                {showLabel && (
                    <span className="rec-label">
                        {isRecording ? 'Recording' : 'Record'}
                    </span>
                )}
            </button>

            {isRecording && (
                <div className="rec-timer">
                    <RecordingTimer />
                </div>
            )}
        </div>
    )
}

const RecordingTimer = () => {
    const [time, setTime] = React.useState(0)

    React.useEffect(() => {
        const interval = setInterval(() => {
            setTime(t => t + 1)
        }, 1000)
        return () => clearInterval(interval)
    }, [])

    const minutes = Math.floor(time / 60)
    const seconds = time % 60

    return (
        <span className="timer-display">
            {String(minutes).padStart(2, '0')}:{String(seconds).padStart(2, '0')}
        </span>
    )
}
