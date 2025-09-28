import asyncio
import io
import json
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import yfinance as yf
from openai import OpenAI
from agents import Agent, Runner

# ---------------- CONFIG ----------------
VOICE = "alloy"    
SAMPLE_RATE = 16000  
RECORD_SECONDS = 2.5
CHANNELS = 1

# ---------------- LOAD CONFIG ----------------
def load_config():
    if os.environ.get("OPENAI_API_KEY"):
        return
    with open("config.json") as f:
        key = json.load(f).get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY in config.json")
    os.environ["OPENAI_API_KEY"] = key

# ---------------- AUDIO HELPERS ----------------
def play_beep():
    """Quick beep to signal recording start."""
    dur = 0.15
    freq = 1000
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
    tone = np.sin(freq * 2 * np.pi * t).astype(np.float32)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()

async def record_audio(seconds=RECORD_SECONDS):
    print("Listening...")
    play_beep()
    audio = await asyncio.to_thread(
        sd.rec,
        int(seconds * SAMPLE_RATE),
        SAMPLE_RATE,
        CHANNELS,
        "float32"
    )
    sd.wait()
    print("Got it.")
    return audio

def say(text: str, client: OpenAI):
    if not text:
        return
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=VOICE,
        input=text,
        response_format="wav",
    )
    data, sr = sf.read(io.BytesIO(response.read()), dtype="float32")
    sd.play(data, sr)
    sd.wait()

# ---------------- LOGIC ----------------
def get_stock_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        fi = dict(getattr(t, "fast_info", {}) or {})
        return {"ticker": ticker, **fi}
    except Exception:
        return {"ticker": ticker, "error": "No data"}

def find_ticker(text: str):
    """Quick heuristic: extract common tickers."""
    lookup = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
        "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
        "nvidia": "NVDA", "netflix": "NFLX", "amd": "AMD", "intel": "INTC"
    }
    s = text.lower()
    for k, v in lookup.items():
        if k in s:
            return v
    import re
    m = re.search(r"\b[A-Z]{1,5}\b", text.upper())
    if m:
        return m.group(0)
    return None

# ---------------- MAIN ----------------
async def main():
    load_config()
    client = OpenAI()

    agent = Agent(
        name="StockAssistant",
        instructions=(
            "You are a quick-fire stock summarizer. "
            "Give 3–5 short facts: price, day change, market cap, 52-week range, and trend (up/down/flat). "
            "Be clear, data-first, and avoid jargon. "
            "End with: 'Not financial advice.'"
        ),
        model="gpt-4.1-mini",
    )

    greeting = "Hi, I’m your stock assistant. Say a company name or ticker!"
    print(greeting)
    say(greeting, client)

    while True:
        cmd = input("Press Enter to talk, or 'q' to quit: ").strip().lower()
        if cmd == "q":
            break

        audio_np = await record_audio()

        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)

        print("Transcribing...")
        try:
            t = client.audio.transcriptions.create(model="whisper-1", file=("input.wav", buf.read(), "audio/wav"))
            user_text = t.text.strip()
        except Exception as e:
            print("[Transcription error]", e)
            say("Sorry, I didn’t catch that.", client)
            continue

        if not user_text:
            say("Could you repeat that?", client)
            continue

        print(f"[You]: {user_text}")

        ticker = find_ticker(user_text)
        if ticker:
            data = get_stock_data(ticker)
            prompt = f"DATA:\n{json.dumps(data)}"
        else:
            prompt = user_text

        result = await Runner.run(agent, prompt)
        reply = (result.final_output or "").strip()

        print(f"[Assistant]: {reply}")
        say(reply, client)

# ---------------- RUN ----------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
