/**
 * useVisionWebSocket Hook
 * Manages WebSocket connection for real-time ASL letter recognition
 */
import { useState, useEffect, useCallback, useRef } from 'react'

export const useVisionWebSocket = ({ onLetterUpdate, onError }) => {
    const [ws, setWs] = useState(null)
    const [isConnected, setIsConnected] = useState(false)
    const reconnectTimeoutRef = useRef(null)

    useEffect(() => {
        const connect = () => {
            try {
                const socket = new WebSocket('ws://localhost:8000/ws/vision')

                socket.onopen = () => {
                    console.log('Vision WebSocket connected')
                    setIsConnected(true)
                }

                socket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data)

                        if (data.type === 'letter_update' || data.type === 'no_hand') {
                            onLetterUpdate?.(data)
                        } else if (data.type === 'buffer_cleared' || data.type === 'letter_deleted') {
                            onLetterUpdate?.(data)
                        } else if (data.type === 'interrupt_ready') {
                            onLetterUpdate?.(data)
                        } else if (data.type === 'error') {
                            onError?.(data.message)
                        }
                    } catch (err) {
                        console.error('Failed to parse vision message:', err)
                    }
                }

                socket.onerror = (error) => {
                    console.error('Vision WebSocket error:', error)
                    onError?.('WebSocket connection error')
                }

                socket.onclose = () => {
                    console.log('Vision WebSocket disconnected')
                    setIsConnected(false)
                    setWs(null)

                    // Auto-reconnect after 3 seconds
                    reconnectTimeoutRef.current = setTimeout(() => {
                        console.log('Attempting to reconnect...')
                        connect()
                    }, 3000)
                }

                setWs(socket)
            } catch (err) {
                console.error('Failed to create WebSocket:', err)
                onError?.('Failed to connect to vision service')
            }
        }

        connect()

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current)
            }
            if (ws) {
                ws.close()
            }
        }
    }, [])

    const sendFrame = useCallback((base64Image) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'frame',
                image: base64Image
            }))
        }
    }, [ws])

    const clearBuffer = useCallback(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'clear_buffer' }))
        }
    }, [ws])

    const deleteLast = useCallback(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'delete_last' }))
        }
    }, [ws])

    const sendInterrupt = useCallback(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'send_interrupt' }))
        }
    }, [ws])

    return {
        sendFrame,
        clearBuffer,
        deleteLast,
        sendInterrupt,
        isConnected
    }
}
