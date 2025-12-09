"""
TopMediai TTS Service

TopMediai Text-to-Speech integration using Playwright to automate web interface.
Falls back to Google Cloud TTS automatically.

Voice IDs (from https://www.topmediai.com/voice-store/):
- Rick: 67ad973f-5d4b-11ee-a861-00163e2ac61b (Rick Sanchez)
- Morty: 67ada016-5d4b-11ee-a861-00163e2ac61b (Morty Smith)

Usage:
    No API key needed - uses web automation
    Falls back to Google Cloud TTS if TopMediai fails
    
Example:
    audio = generate_topmediai_tts(
        text="Hello world",
        voice_id="67ada016-5d4b-11ee-a861-00163e2ac61b",
        fallback_to_google=True,
        google_voice_id="en-US-Wavenet-D"
    )
"""

import os
import requests
import io
import asyncio
from typing import Optional
import time

# Try to import playwright
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Install with: pip install playwright && playwright install chromium")


class TopMediaiTTS:
    """TopMediai TTS client with web automation and API fallback."""
    
    # TopMediai API endpoint (inferred from web interface)
    BASE_URL = "https://api.topmediai.com/v1/tts"  # May need adjustment
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TopMediai TTS client.
        
        Args:
            api_key: TopMediai API key (optional - web automation doesn't need it)
        """
        self.api_key = api_key or os.getenv("TOPMEDIAI_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        
        if not PLAYWRIGHT_AVAILABLE:
            print("Warning: Playwright not available. TopMediai web automation disabled.")
            print("Install with: pip install playwright && playwright install chromium")
    
    def generate(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        volume: float = 0.5,
        pitch: float = 0.5,
    ) -> Optional[bytes]:
        """
        Generate speech using TopMediai TTS.
        
        Args:
            text: Text to convert to speech
            voice_id: TopMediai voice ID
            speed: Speech speed (0.5-2.0)
            volume: Volume (0.0-1.0)
            pitch: Pitch adjustment (0.0-1.0)
            
        Returns:
            Audio bytes (MP3) or None if failed
        """
        try:
            # Method 1: Try web interface automation (primary method)
            if PLAYWRIGHT_AVAILABLE:
                print(f"Trying TopMediai web automation for voice {voice_id}...")
                response = self._try_web_interface(text, voice_id, speed, volume, pitch)
                if response:
                    print("TopMediai web automation succeeded!")
                    return response
                print("TopMediai web automation failed, trying API...")
            
            # Method 2: Try direct API call (fallback)
            response = self._try_api_call(text, voice_id, speed, volume, pitch)
            if response:
                print("TopMediai API call succeeded!")
                return response
            
            print("All TopMediai methods failed")
            return None
            
        except Exception as e:
            print(f"TopMediai TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _try_api_call(
        self,
        text: str,
        voice_id: str,
        speed: float,
        volume: float,
        pitch: float,
    ) -> Optional[bytes]:
        """Try API endpoint method."""
        try:
            # TopMediai might use different endpoints - try common patterns
            endpoints = [
                "https://api.topmediai.com/v1/tts/generate",
                "https://api.topmediai.com/v1/voiceover/generate",
                "https://www.topmediai.com/api/tts",
                "https://www.topmediai.com/api/v1/tts/generate",
            ]
            
            # Try with different payload formats
            payloads = [
                {
                    "text": text,
                    "voice_id": voice_id,
                    "speed": speed,
                    "volume": volume,
                    "pitch": pitch,
                    "format": "mp3",
                },
                {
                    "input": text,
                    "voice": voice_id,
                    "speed": int(speed * 100),
                    "volume": int(volume * 100),
                },
                {
                    "text": text,
                    "voiceId": voice_id,
                    "speed": speed,
                    "volume": volume,
                },
            ]
            
            for endpoint in endpoints:
                for payload in payloads:
                    try:
                        resp = self.session.post(
                            endpoint,
                            json=payload,
                            timeout=15,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                                "Referer": "https://www.topmediai.com/app/text-to-speech/",
                            }
                        )
                        
                        if resp.status_code == 200:
                            # Check if response is audio
                            content_type = resp.headers.get("content-type", "")
                            if "audio" in content_type:
                                return resp.content
                            
                            # Try parsing as JSON
                            try:
                                data = resp.json()
                                if "audio_url" in data or "url" in data:
                                    audio_url = data.get("audio_url") or data.get("url")
                                    audio_resp = requests.get(audio_url, timeout=10)
                                    if audio_resp.status_code == 200:
                                        return audio_resp.content
                                if "audio" in data or "data" in data:
                                    import base64
                                    audio_data = data.get("audio") or data.get("data")
                                    if isinstance(audio_data, str):
                                        return base64.b64decode(audio_data)
                            except:
                                pass
                    except requests.exceptions.RequestException:
                        continue
                    except Exception as e:
                        print(f"TopMediai endpoint {endpoint} error: {e}")
                        continue
            
            return None
            
        except Exception as e:
            print(f"TopMediai API call failed: {e}")
            return None
    
    def _try_web_interface(
        self,
        text: str,
        voice_id: str,
        speed: float,
        volume: float,
        pitch: float,
    ) -> Optional[bytes]:
        """Try web interface method using Playwright automation."""
        if not PLAYWRIGHT_AVAILABLE:
            return None
        
        try:
            # Run async playwright in sync context
            return asyncio.run(self._automate_topmediai_web(text, voice_id, speed, volume, pitch))
        except Exception as e:
            print(f"TopMediai web interface failed: {e}")
            return None
    
    async def _automate_topmediai_web(
        self,
        text: str,
        voice_id: str,
        speed: float,
        volume: float,
        pitch: float,
    ) -> Optional[bytes]:
        """Automate TopMediai web interface using Playwright."""
        try:
            async with async_playwright() as p:
                # Launch browser in headless mode
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = await context.new_page()
                
                # Set up audio capture from network requests
                audio_data = None
                audio_captured = asyncio.Event()
                
                async def handle_response(response):
                    """Capture audio from network responses"""
                    nonlocal audio_data
                    content_type = response.headers.get('content-type', '')
                    url = response.url
                    
                    # Check if this is an audio file
                    if ('audio' in content_type or 
                        url.endswith(('.mp3', '.wav', '.ogg', '.m4a')) or
                        'audio' in url.lower()):
                        try:
                            body = await response.body()
                            if body and len(body) > 1000:  # Audio files are usually > 1KB
                                audio_data = body
                                audio_captured.set()
                                print(f"✅ Captured audio from network: {len(body)} bytes from {url}")
                        except Exception as e:
                            print(f"Error capturing audio: {e}")
                
                page.on('response', handle_response)
                
                # Navigate to TopMediai TTS page with voice ID
                url = f"https://www.topmediai.com/app/text-to-speech/?voice={voice_id}"
                print(f"Navigating to TopMediai: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for page to load and check if we need to login
                await page.wait_for_timeout(5000)  # Increased wait time
                
                # Check for login prompts and skip if possible
                try:
                    close_btn = await page.query_selector('.close, [aria-label="Close"], button:has-text("×")')
                    if close_btn:
                        await close_btn.click()
                        await page.wait_for_timeout(1000)
                except:
                    pass
                
                # Find and fill the text input
                # The textarea might have different selectors - try common ones
                text_selectors = [
                    'textarea[placeholder*="text"]',
                    'textarea[placeholder*="Text"]',
                    'textarea[placeholder*="请输入"]',  # Chinese placeholder
                    'textarea',
                    'input[type="text"]',
                    '[contenteditable="true"]',
                    '#text-input',
                    '.text-input',
                ]
                
                text_filled = False
                for selector in text_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            # Clear first
                            await element.click()
                            await element.fill('')
                            await element.fill(text)
                            text_filled = True
                            print(f"Filled text using selector: {selector}")
                            break
                    except Exception as e:
                        print(f"Selector {selector} failed: {e}")
                        continue
                
                if not text_filled:
                    print("Could not find text input on TopMediai page")
                    # Take screenshot for debugging
                    try:
                        await page.screenshot(path="/tmp/topmediai_debug.png")
                        print("Screenshot saved to /tmp/topmediai_debug.png")
                    except:
                        pass
                    await browser.close()
                    return None
                
                # Set speed/volume/pitch if controls exist
                try:
                    # Speed slider (1x = 100%)
                    speed_input = await page.query_selector('input[type="range"][aria-label*="Speed"], input[type="range"]')
                    if speed_input:
                        speed_value = int(speed * 100)
                        await speed_input.fill(str(speed_value))
                except:
                    pass
                
                # Click generate button
                generate_selectors = [
                    'button:has-text("Generate")',
                    'button:has-text("生成")',
                    'button:has-text("Generate Speech")',
                    'button[type="submit"]',
                    '.generate-btn',
                    '#generate-btn',
                    'button.btn-primary',
                    'button.primary',
                ]
                
                generate_clicked = False
                for selector in generate_selectors:
                    try:
                        button = await page.query_selector(selector)
                        if button:
                            # Scroll into view
                            await button.scroll_into_view_if_needed()
                            await page.wait_for_timeout(500)
                            await button.click()
                            generate_clicked = True
                            print(f"Clicked generate using selector: {selector}")
                            break
                    except Exception as e:
                        print(f"Generate selector {selector} failed: {e}")
                        continue
                
                if not generate_clicked:
                    print("Could not find generate button, trying Enter key")
                    # Try pressing Enter on the textarea
                    try:
                        textarea = await page.query_selector('textarea')
                        if textarea:
                            await textarea.press('Enter')
                            generate_clicked = True
                    except:
                        pass
                
                if not generate_clicked:
                    print("Could not trigger generation")
                    await browser.close()
                    return None
                
                # Wait for audio to generate - check if we already captured it from network
                try:
                    # Wait for audio capture (with timeout)
                    try:
                        await asyncio.wait_for(audio_captured.wait(), timeout=30.0)
                        if audio_data:
                            await browser.close()
                            return audio_data
                    except asyncio.TimeoutError:
                        print("Timeout waiting for audio from network, trying DOM methods...")
                    
                    # Fallback: Wait for audio element to appear
                    try:
                        await page.wait_for_selector('audio', timeout=10000)
                    except:
                        pass
                    
                    # Method 1: Check for audio element src
                    audio_url = None
                    audio_element = await page.query_selector('audio')
                    if audio_element:
                        audio_url = await audio_element.get_attribute('src')
                        if audio_url:
                            print(f"Found audio URL from element: {audio_url}")
                    
                    # Method 2: Check for download link
                    if not audio_url:
                        download_link = await page.query_selector('[download], .download-btn, a[href*=".mp3"], a[href*=".wav"]')
                        if download_link:
                            audio_url = await download_link.get_attribute('href')
                            print(f"Found audio URL from download link: {audio_url}")
                    
                    if audio_url:
                        # Download the audio
                        if audio_url.startswith('/'):
                            audio_url = f"https://www.topmediai.com{audio_url}"
                        elif not audio_url.startswith('http'):
                            audio_url = f"https://www.topmediai.com/{audio_url}"
                        
                        audio_response = await page.request.get(audio_url)
                        if audio_response.status == 200:
                            audio_bytes = await audio_response.body()
                            await browser.close()
                            return audio_bytes
                    
                    # Method 3: Try to get base64 from page
                    audio_src = await page.evaluate("""
                        () => {
                            const audio = document.querySelector('audio');
                            if (audio && audio.src) {
                                return audio.src;
                            }
                            return null;
                        }
                    """)
                    
                    if audio_src and audio_src.startswith('data:audio'):
                        import base64
                        # Extract base64 data
                        header, data = audio_src.split(',', 1)
                        audio_bytes = base64.b64decode(data)
                        await browser.close()
                        return audio_bytes
                    
                    print("⚠️ Could not find audio after generation")
                    
                except Exception as e:
                    print(f"Error waiting for audio: {e}")
                    import traceback
                    traceback.print_exc()
                
                await browser.close()
                return None
                
        except Exception as e:
            print(f"Playwright automation error: {e}")
            import traceback
            traceback.print_exc()
            return None


def generate_topmediai_tts(
    text: str,
    voice_id: str,
    fallback_to_google: bool = True,
    google_voice_id: Optional[str] = None,
    speed: float = 1.0,
) -> Optional[bytes]:
    """
    Generate TTS using TopMediai with Google Cloud fallback.
    
    Args:
        text: Text to convert
        voice_id: TopMediai voice ID
        fallback_to_google: Whether to fallback to Google TTS
        google_voice_id: Google voice ID for fallback
        speed: Speech speed multiplier (0.5-2.0, default 1.0)
        
    Returns:
        Audio bytes or None
    """
    # Clamp speed to valid range
    speed = max(0.5, min(2.0, speed))
    
    # Try TopMediai first (with timeout)
    if PLAYWRIGHT_AVAILABLE:
        print(f"🎙️ Attempting TopMediai TTS for voice {voice_id} at speed {speed}x...")
        try:
            topmediai = TopMediaiTTS()
            # Add timeout wrapper
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("TopMediai TTS timeout")
            
            # Try TopMediai with 20 second timeout
            audio = None
            try:
                # Run with timeout - pass speed parameter
                audio = topmediai.generate(text, voice_id, speed=speed)
            except Exception as e:
                print(f"TopMediai error: {e}")
            
            if audio and len(audio) > 0:
                print(f"✅ TopMediai TTS success: {len(audio)} bytes")
                return audio
            else:
                print("❌ TopMediai returned no audio")
        except Exception as e:
            print(f"TopMediai failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback to Google Cloud
    if fallback_to_google:
        try:
            from google.cloud import texttospeech
            
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                print("⚠️ Google TTS not configured (GOOGLE_APPLICATION_CREDENTIALS not set)")
                return None
            
            print(f"🔄 Falling back to Google Cloud TTS (voice: {google_voice_id or 'en-US-Wavenet-D'})")
            client = texttospeech.TextToSpeechClient()
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=google_voice_id or "en-US-Wavenet-D"
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            print(f"✅ Google Cloud TTS success: {len(response.audio_content)} bytes")
            return response.audio_content
            
        except Exception as e:
            print(f"❌ Google Cloud fallback also failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None


