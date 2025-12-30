#!/usr/bin/env python3
import os
import json
import csv
import asyncio
import base64
import logging
import websockets
import httpx
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from openai import AsyncOpenAI
from dotenv import load_dotenv # Recommended for local dev

# =========================
# CONFIG & SETUP
# =========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
PUBLIC_HOST = os.getenv("PUBLIC_HOST")

# Validation
if not all([OPENAI_API_KEY, DEEPGRAM_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]):
    raise ValueError("Missing required environment variables. Check your .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Global Async Clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
# We will create httpx clients per session or use a shared one with proper lifespan management
# For simplicity in this script, we use a global one but close it on shutdown.
http_client = httpx.AsyncClient(timeout=30.0)

# State Management
active_sessions: Dict[str, "CallSession"] = {}
sessions_lock = asyncio.Lock()

# =========================
# SESSION CLASS
# =========================
class CallSession:
    def __init__(self, name: str, phone: str, call_sid: str):
        self.name = name
        self.phone = phone
        self.call_sid = call_sid
        self.is_speaking = False
        self.tts_task: Optional[asyncio.Task] = None
        self.dg_ws: Optional[websockets.WebSocketClientProtocol] = None
        
        # Load script content
        script_content = """You are Emma, a friendly, confident cold-calling agent representing a marketing company.

THE OFFER:
We run targeted SMS campaigns to homeowners in your service area (15-20 mile radius). We ask homeowners if they need roof repairs, replacements, or storm damage checks. If interested, we qualify them and send you the homeowner details. You only pay for qualified leads at $90 each. There's a $600 monthly retainer covering software, data, SMS tools, and everything on our end.

KEY POINTS:
- Homeowners from their exact service area
- Leads are exclusive
- Pay only for qualified leads
- Retainer covers tools + software
- No long-term contract (month-to-month)

CALL FLOW:
1. Opener: "Hey, is this the owner? Great — this is Emma calling from a marketing agency. We help roofing companies get exclusive homeowner leads without relying on HomeAdvisor or shared platforms."

2. Value Pitch: "We use SMS outreach to connect with homeowners in your service area who need roof repair or replacement. You only pay for qualified leads at $90, plus a $600 retainer for the software."

QUALIFICATION QUESTIONS (if interested):
- Which areas or ZIP codes do you usually serve?
- Do you prefer repair jobs, replacement jobs, or both?
- How soon can you take on new clients?
- Do you handle insurance-related storm damage jobs?

OBJECTION HANDLING:
- "Too expensive" → "I hear you. Most roofers feel that way at first. But because you only pay for qualified leads, the ROI is strong — even closing one job usually covers the entire month."
- "Already have marketing" → "That's great — this doesn't replace anything. It just adds exclusive, local homeowner leads on top of what you're already doing."
- "Don't trust SMS" → "Totally understand. The thing is — homeowners respond to SMS faster than calls or emails. It's non-intrusive and gives you higher reply rates."
- "Send me info" → "For sure. Before I send anything, it's better to give you a quick 5-minute explanation so you can see exactly how leads are qualified. When works better, later today or tomorrow?"
- "Too many unqualified leads" → "That's exactly why our system is different. You pay only for qualified interest — we filter out renters, tire kickers, and people outside your area."
- "Tried agencies before" → "Totally get it. Our model is different — no contracts, no upfront packages, and no paying for junk leads."
- "Too busy" → "No worries. Even a 5-minute call can show you if this can increase your deal flow without adding work for you."
- "Not interested" → "Totally fine. Before I hop off — are you currently taking on new roofing jobs, or is your schedule full?"

CLOSING GOAL:
Book a 5-10 minute demo with the owner: "Let's set up a quick 5-10 minute call so I can show you how many homeowners in your exact area are responding. What time works better — today or tomorrow?"

TONE GUIDELINES:
- Speak casually, naturally, friendly
- Keep replies VERY SHORT (1-2 sentences max)
- Don't sound like reading a script
- Never argue or oversell
- Lead the conversation gently
- Always push toward a follow-up call
- Be conversational and human-like"""
        
        self.history = [
            {"role": "system", "content": script_content}
        ]

    async def get_reply(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        # Trim history to save tokens but keep context
        if len(self.history) > 12:
            # Keep system message and last 10 exchanges
            self.history = [self.history[0]] + self.history[-10:]

        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.history,
                max_tokens=60,  # Keep responses short
                temperature=0.8,
            )
            reply = resp.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            return "Sorry, could you repeat that?"

    async def stop_speaking(self):
        """Cancels current TTS task if playing."""
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
        self.is_speaking = False

    async def close(self):
        """Clean up resources."""
        await self.stop_speaking()
        if self.dg_ws:
            try:
                await self.dg_ws.close()
            except Exception:
                pass

# =========================
# HELPERS
# =========================
async def send_mark(ws: WebSocket, stream_sid: str):
    """Send a mark message to sync Twilio media stream."""
    await ws.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": "responseEnd"}})

async def send_clear(ws: WebSocket, stream_sid: str):
    """Tell Twilio to clear its audio buffer (stop playing immediately)."""
    await ws.send_json({"event": "clear", "streamSid": stream_sid})

async def text_to_speech_stream(session: CallSession, text: str, ws: WebSocket, stream_sid: str):
    """
    Fetches TTS audio and streams it to Twilio at the correct playback rate.
    8kHz * 1 byte (mulaw) = 8000 bytes per second.
    """
    session.is_speaking = True
    
    try:
        # 1. Fetch Audio from Deepgram with consistent female voice
        r = await http_client.post(
            "https://api.deepgram.com/v1/speak",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json"
            },
            params={
                "model": "aura-luna-en",  # Luna voice
                "encoding": "mulaw",
                "sample_rate": 8000,
                "container": "none",  # Raw audio, no container
            },
            json={"text": text},
            timeout=10.0
        )
        r.raise_for_status()
        audio_data = r.content
        
        if len(audio_data) == 0:
            logger.warning("Empty audio response from Deepgram")
            return
        
        # 2. Stream to Twilio with proper timing
        # Use 160 bytes (20ms chunks) for clearer, smoother playback
        chunk_size = 160
        bytes_per_second = 8000
        
        for i in range(0, len(audio_data), chunk_size):
            if not session.is_speaking: 
                break

            chunk = audio_data[i:i + chunk_size]
            await ws.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode()},
            })
            
            # Calculate precise sleep time for this chunk
            chunk_duration = len(chunk) / bytes_per_second
            await asyncio.sleep(chunk_duration)

        await send_mark(ws, stream_sid)

    except asyncio.CancelledError:
        # Send clear event immediately upon cancellation
        await send_clear(ws, stream_sid)
        logger.info(f"TTS Cancelled for call {session.call_sid}")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
    finally:
        session.is_speaking = False

# =========================
# WEBSOCKET MEDIA STREAM
# =========================
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    session: Optional[CallSession] = None
    stream_sid = None
    
    # Define callback logic before passing to Deepgram
    async def on_deepgram_message(data: dict):
        nonlocal session, stream_sid
        
        msg_type = data.get("type")
        
        # BARGE-IN LOGIC: Use SpeechStarted event for instant reaction
        if msg_type == "SpeechStarted":
            if session and session.is_speaking:
                logger.info("User interrupted (VAD).")
                await session.stop_speaking()
            return

        if msg_type == "Results":
            is_final = data.get("is_final", False)
            transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
            
            if not transcript:
                return

            if is_final:
                logger.info(f"User said: {transcript}")
                # If user spoke while bot was finishing up, ensure we stop
                if session.is_speaking:
                    await session.stop_speaking()
                
                # Generate Reply
                reply = await session.get_reply(transcript)
                logger.info(f"Emma says: {reply}")
                
                # Start Speaking
                session.tts_task = asyncio.create_task(text_to_speech_stream(session, reply, ws, stream_sid))

    try:
        while True:
            message = await ws.receive_text()
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["customParameters"]["CallSid"]
                
                # Retrieve session safely
                async with sessions_lock:
                    session = active_sessions.get(call_sid)
                
                if not session:
                    logger.error(f"Session {call_sid} not found!")
                    await ws.close()
                    break

                # Connect to Deepgram
                try:
                    session.dg_ws = await websockets.connect(
                        "wss://api.deepgram.com/v1/listen?"
                        "model=nova-2&encoding=mulaw&sample_rate=8000"
                        "&interim_results=false&vad_events=true",
                        additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                    )
                    
                    # Start Deepgram Listener Loop
                    async def dg_listener():
                        try:
                            async for msg in session.dg_ws:
                                await on_deepgram_message(json.loads(msg))
                        except websockets.exceptions.ConnectionClosed:
                            logger.info("Deepgram connection closed")
                        except Exception as e:
                            logger.error(f"Deepgram Error: {e}")

                    asyncio.create_task(dg_listener())
                
                except Exception as e:
                    logger.error(f"Failed to connect to Deepgram: {e}")
                    await ws.close()
                    break

            elif event == "media" and session and session.dg_ws:
                # Forward raw audio from Twilio to Deepgram
                audio_payload = data["media"]["payload"]
                await session.dg_ws.send(base64.b64decode(audio_payload))

            elif event == "stop":
                logger.info(f"Twilio stream stop event received.")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        # Cleanup
        if session:
            async with sessions_lock:
                # Optional: Remove from active_sessions immediately or keep for history
                # active_sessions.pop(session.call_sid, None)
                await session.close()
        logger.info("Media stream handler finished.")

# =========================
# TWILIO WEBHOOKS
# =========================
@app.post("/twiml")
async def twiml(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    
    async with sessions_lock:
        session = active_sessions.get(call_sid)

    if not session:
        logger.error(f"TwiML request for unknown CallSid: {call_sid}")
        return Response(content="Error", status_code=404)

    response = VoiceResponse()
    response.say(f"Hi {session.name}, this is Emma.")
    
    connect = Connect()
    stream = Stream(url=f"wss://{PUBLIC_HOST}/media-stream")
    stream.parameter(name="CallSid", value=call_sid)
    connect.append(stream)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")

# =========================
# CAMPAIGN LOGIC
# =========================
def read_leads(filename="leads.csv"):
    if not os.path.exists(filename):
        logger.warning(f"Leads file {filename} not found. Returning empty list.")
        return []
    with open(filename) as f:
        return list(csv.DictReader(f))

async def call_lead(lead: dict):
    try:
        # Twilio client is synchronous, run in thread pool to not block event loop
        call = await asyncio.to_thread(
            twilio_client.calls.create,
            to=lead["phone"],
            from_=TWILIO_PHONE_NUMBER,
            url=f"https://{PUBLIC_HOST}/twiml",
            status_callback=f"https://{PUBLIC_HOST}/status-callback", # Recommended for cleanup
            status_callback_event="completed"
        )
        
        logger.info(f"Initiated call to {lead['name']} at {lead['phone']}. SID: {call.sid}")
        
        async with sessions_lock:
            active_sessions[call.sid] = CallSession(lead["name"], lead["phone"], call.sid)
            
    except Exception as e:
        logger.error(f"Failed to call {lead['phone']}: {e}")

@app.post("/start-campaign")
async def start_campaign():
    leads = read_leads()
    if not leads:
        return {"status": "error", "message": "No leads found"}
        
    for lead in leads:
        await call_lead(lead)
        await asyncio.sleep(10) # Stagger calls slightly
    return {"status": "started", "calls_queued": len(leads)}

@app.post("/status-callback")
async def status_callback(request: Request):
    """Clean up session when call ends."""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    
    if call_status == "completed" or call_status == "failed":
        async with sessions_lock:
            session = active_sessions.pop(call_sid, None)
            if session:
                await session.close()
    
    return Response(status_code=200)

# =========================
# LIFECYCLE
# =========================
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)