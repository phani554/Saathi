#!/usr/bin/env python3
"""
Saathi — Voice-First WhatsApp Companion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMARY   : Gemini 3.1 Flash-Lite via agno.models.google.Gemini
FALLBACK  : LFM2.5-1.2B Q4_K_M via agno.models.llama_cpp.LlamaCpp
            → llama-server (OpenAI-compat REST, port LLAMA_PORT)
            → Agno handles tool calling, memory, session — identical
              to the cloud path, not a degraded custom loop

SHARED    : BOTH agents share the same SQLite file (saathi.db) for:
            • SqliteMemoryDb   — persists extracted user facts
            • SqliteAgentStorage — persists session history
            Memories learned in cloud mode survive a local fallback
            and vice versa.

VOICE     : Sarvam Saaras v3 STT (codemix → Hinglish-native)
            Sarvam Bulbul  v3 TTS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY agno.models.llama_cpp.LlamaCpp instead of raw OpenAI client?

  LlamaCpp is Agno's first-class subclass of OpenAILike. It means:
  • Agno's full memory stack (Memory v2 + SqliteMemoryDb) works
  • Agno's SqliteAgentStorage works — sessions persist across reboots
  • Agno registers Python functions as tools natively — no manual
    schema writing, no manual tool_calls parsing
  • Both paths use identical Agent(...) construction — one code path
  • llama-server+--jinja exposes LFM2.5's native chat template, so
    <|tool_call_start|> tokens are handled by the server, not us

WHY --jinja on llama-server?
  Without --jinja the server uses a generic template and LFM2.5's
  native Pythonic function-call tokens are not injected, which
  degrades tool-call reliability. --jinja makes llama-server render
  the model's own Jinja chat template — the same one Liquid AI
  trained the model with.

WHY Q4_K_M GGUF?
  • ~750 MB RAM vs ~2.4 GB for BF16 PyTorch
  • llama.cpp CPU decode: 113–239 tok/s vs ~15 tok/s via torch
  • Zero torch/transformers in fallback path — cold start is instant
"""

import base64
import gc
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── PATHS ─────────────────────────────────────────────────────────────────────
AGENT_DIR      = Path(__file__).parent.resolve()
VCARD_PATH     = AGENT_DIR / "contacts.vcf"
NICKNAMES_PATH = AGENT_DIR / "nicknames.json"
MEMORY_DB      = str(AGENT_DIR / "saathi.db")   # SHARED by both agents
GGUF_DIR       = AGENT_DIR / "models"
GGUF_FILE      = GGUF_DIR / "lfm2.5-1.2b-instruct-q4_k_m.gguf"

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BRIDGE_URL   = os.getenv("BRIDGE_URL", "http://localhost:3000")
LLAMA_PORT   = int(os.getenv("LLAMA_PORT", "8082"))
LLAMA_CTX    = int(os.getenv("LLAMA_CTX", "8192"))
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
LFM_HF_REPO  = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
LFM_HF_FILE  = "lfm2.5-1.2b-instruct-q4_k_m.gguf"

# ─── SARVAM VOICE ─────────────────────────────────────────────────────────────
SARVAM_KEY      = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "codemix")  # codemix = Hinglish
SARVAM_STT_URL  = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL  = "https://api.sarvam.ai/text-to-speech"
VOICE_READY     = bool(SARVAM_KEY)
SAMPLE_RATE     = 16000
RECORD_SECS     = int(os.getenv("RECORD_SECS", "7"))

# ──────────────────────────────────────────────────────────────────────────────
#  VOICE LAYER
# ──────────────────────────────────────────────────────────────────────────────

def record_audio(duration: int = RECORD_SECS) -> Optional[str]:
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        print(f"🎤  Recording {duration}s … speak now")
        frames = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                        channels=1, dtype=np.int16)
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AGENT_DIR)
        wavfile.write(tmp.name, SAMPLE_RATE, frames)
        tmp.close()
        return tmp.name
    except Exception as exc:
        print(f"⚠️  Record failed: {exc}")
        return None


def sarvam_stt(audio_path: str) -> Optional[str]:
    """Sarvam Saaras v3 — codemix mode handles Hinglish/mixed-language natively."""
    if not SARVAM_KEY:
        return None
    try:
        with open(audio_path, "rb") as fh:
            resp = requests.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": SARVAM_KEY},
                files={"file": ("audio.wav", fh, "audio/wav")},
                data={"model": "saaras:v3", "mode": SARVAM_STT_MODE},
                timeout=30,
            )
        resp.raise_for_status()
        body = resp.json()
        transcript = (body.get("transcript") or "").strip()
        lang = body.get("language_code", "?")
        if transcript:
            print(f"📝  [{lang}/{SARVAM_STT_MODE}] {transcript}")
        return transcript or None
    except Exception as exc:
        print(f"⚠️  STT error: {exc}")
        return None
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def _detect_lang(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F: return "hi-IN"
        if 0x0C00 <= cp <= 0x0C7F: return "te-IN"
        if 0x0B80 <= cp <= 0x0BFF: return "ta-IN"
        if 0x0980 <= cp <= 0x09FF: return "bn-IN"
        if 0x0A80 <= cp <= 0x0AFF: return "gu-IN"
        if 0x0C80 <= cp <= 0x0CFF: return "kn-IN"
        if 0x0D00 <= cp <= 0x0D7F: return "ml-IN"
        if 0x0A00 <= cp <= 0x0A7F: return "pa-IN"
    return "en-IN"


def sarvam_tts(text: str) -> Optional[bytes]:
    """Sarvam Bulbul v3 — auto-detects language from Unicode script."""
    if not SARVAM_KEY or not text.strip():
        return None
    lang = _detect_lang(text)
    try:
        resp = requests.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"},
            json={
                "inputs": [text[:2500]],
                "target_language_code": lang,
                "model": "bulbul:v3",
                "speaker": "meera",
                "pace": 0.85,
                "enable_preprocessing": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        audios = resp.json().get("audios", [])
        return base64.b64decode(audios[0]) if audios else None
    except Exception as exc:
        print(f"⚠️  TTS error: {exc}")
        return None


def play_audio(wav_bytes: bytes) -> None:
    if not wav_bytes:
        return
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        rate, data = wavfile.read(io.BytesIO(wav_bytes))
        if data.ndim > 1:
            data = data[:, 0]
        sd.play(data.astype(np.float32) / 32768.0, rate)
        sd.wait()
    except Exception as exc:
        print(f"⚠️  Playback failed: {exc}")


def speak(text: str) -> None:
    if VOICE_READY:
        play_audio(sarvam_tts(text))


def voice_input() -> Optional[str]:
    path = record_audio()
    return sarvam_stt(path) if path else None


# ──────────────────────────────────────────────────────────────────────────────
#  TOOL IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_nicknames() -> dict:
    if NICKNAMES_PATH.exists():
        return json.loads(NICKNAMES_PATH.read_text(encoding="utf-8"))
    return {}


def save_nickname(nickname: str, real_name: str) -> str:
    """Permanently save a nickname → real contact name mapping."""
    nicks = load_nicknames()
    nicks[nickname.lower()] = real_name
    NICKNAMES_PATH.write_text(
        json.dumps(nicks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return f"Saved: '{nickname}' → '{real_name}'."


def get_contact_number(name: str) -> str:
    """
    Look up a WhatsApp phone number from contacts.vcf by name.
    Resolves nicknames first. Filters landlines. Handles duplicates.
    """
    try:
        nicks = load_nicknames()
        search_name = nicks.get(name.lower(), name)
        if not VCARD_PATH.exists():
            return "Error: contacts.vcf not found. Place it in the agent/ folder."
        raw = VCARD_PATH.read_text(encoding="utf-8", errors="ignore")
        matches: dict[str, str] = {}
        for card in raw.split("BEGIN:VCARD"):
            if not card.strip():
                continue
            fn = re.search(r"^FN:(.+)$", card, re.MULTILINE)
            if not fn:
                continue
            full_name = fn.group(1).strip()
            if search_name.lower() not in full_name.lower():
                continue
            for raw_tel in re.findall(r"^TEL.*?:(.+)$", card, re.MULTILINE):
                clean = re.sub(r"[^\d+]", "", raw_tel)
                digits = re.sub(r"\D", "", clean)
                if clean.startswith("0") or clean.startswith("+910"):
                    continue
                if len(digits) >= 10:
                    matches[full_name] = clean
                    break
        if not matches:
            suffix = f" (alias for '{name}')" if search_name != name else ""
            return f"Contact '{search_name}'{suffix} not found in contacts."
        if len(matches) == 1:
            return list(matches.values())[0]
        names = ", ".join(matches.keys())
        return f"Multiple contacts found for '{search_name}': {names}. Ask the user which one."
    except Exception as exc:
        return f"Error reading contacts: {exc}"


def get_group_id(group_name: str) -> str:
    """Find a WhatsApp group JID by partial group name."""
    try:
        res = requests.get(f"{BRIDGE_URL}/groups", timeout=5)
        if res.ok:
            for g in res.json().get("groups", []):
                if group_name.lower() in g["name"].lower():
                    return g["id"]
        return f"Group '{group_name}' not found."
    except Exception:
        return "Error: WhatsApp bridge unreachable."


def send_whatsapp_message(identifier: str, message: str) -> str:
    """
    Send a WhatsApp message. ONLY call this after the user has explicitly
    confirmed ('yes', 'send it', 'haan', etc.). Never call preemptively.
    """
    try:
        res = requests.post(
            f"{BRIDGE_URL}/send",
            json={"phone": identifier, "message": message},
            timeout=10,
        )
        return "Message sent successfully." if res.ok else f"Failed: {res.text}"
    except Exception:
        return "Error: WhatsApp bridge unreachable."


def manage_whatsapp_session(action: str) -> str:
    """Control the WhatsApp bridge session. Actions: start | logout | get_qr"""
    try:
        if action == "logout":
            requests.post(f"{BRIDGE_URL}/auth/logout", timeout=5)
            return "Logged out successfully."
        if action == "start":
            requests.post(f"{BRIDGE_URL}/start", timeout=5)
            return "Session starting — fetch QR in ~3 seconds."
        if action == "get_qr":
            res = requests.get(f"{BRIDGE_URL}/auth/qr", timeout=5)
            return res.json().get("qr", "QR not ready.") if res.ok else "QR not ready."
        return f"Unknown action '{action}'. Use: start | logout | get_qr"
    except Exception:
        return "Error: WhatsApp bridge unreachable."


def check_bridge_health() -> str:
    """Check if the WhatsApp bridge is connected and ready to send messages."""
    try:
        data = requests.get(f"{BRIDGE_URL}/health", timeout=3).json()
        if data.get("connected"):
            return "Bridge is connected and ready."
        if data.get("qrPending"):
            return "Bridge needs QR scan. Call manage_whatsapp_session(action='get_qr')."
        return "Bridge is starting up — retry in a moment."
    except Exception:
        return "Bridge unreachable — is the Node.js bridge running?"


# All tools in one list — passed directly to Agent(tools=[...]) in both paths
SAATHI_TOOLS = [
    get_contact_number,
    get_group_id,
    send_whatsapp_message,
    manage_whatsapp_session,
    save_nickname,
    check_bridge_health,
]

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SAATHI_SYSTEM = """\
You are Saathi, a warm, patient WhatsApp assistant for seniors and people with \
accessibility needs. Tone: friendly, clear, never use tech jargon.

LANGUAGE: Always reply in the user's language — Hinglish, Hindi, Telugu, \
Tamil, English, etc.

NICKNAME RULE:
If the user says "message my boss / mom / doctor" and get_contact_number fails, \
ask for their real saved name. Once told, IMMEDIATELY call save_nickname to \
remember it permanently before doing anything else.

MESSAGE SENDING SOP — follow ALL steps, never skip:
  1. GATHER   — Confirm recipient AND exact message text.
  2. LOOKUP   — Call get_contact_number (or get_group_id for groups).
  3. DRAFT    — Read the draft back word-for-word. Ask: "Should I send this?"
  4. WAIT     — Do NOT proceed until user says yes / haan / send it.
  5. SEND     — Only now call send_whatsapp_message.
  6. CONFIRM  — Say "Sent!" clearly.

CRITICAL: Never call send_whatsapp_message without explicit confirmation.\
"""

# ──────────────────────────────────────────────────────────────────────────────
#  SHARED AGNO MEMORY STACK  (identical for both cloud and local agents)
# ──────────────────────────────────────────────────────────────────────────────

def _build_memory_and_storage(model):
    """
    Build Memory + Storage backed by the shared saathi.db SQLite file.

    Both the cloud Gemini agent and the local LFM2.5 agent call this with
    their respective model instances. Since the DB path is the same, all
    learned facts and session history are shared across both modes.

    Memory (agno.memory.v2):
      enable_agentic_memory=True  → the model extracts user facts at the end
      of each run and stores them in SqliteMemoryDb. On the next run, Agno
      injects those facts into the system context automatically.

    Storage (SqliteAgentStorage):
      add_history_to_messages=True + num_history_runs=N → the last N runs are
      loaded from disk and prepended to the context each turn. This gives
      true cross-session conversational continuity without re-prompting.

    Note on 1.2B memory extraction:
      LFM2.5 extracts simple facts ("name is Rahul", "prefers Hindi") very
      reliably. Complex multi-hop reasoning during extraction is not expected.
      The important point: if Gemini already extracted rich memories, they are
      in the DB and LFM2.5 will READ them in context even without re-extracting.
    """
    from agno.memory.v2.memory import Memory
    from agno.memory.v2.db.sqlite import SqliteMemoryDb
    from agno.storage.agent.sqlite import SqliteAgentStorage

    memory = Memory(
        model=model,
        db=SqliteMemoryDb(table_name="saathi_memory", db_file=MEMORY_DB),
    )
    storage = SqliteAgentStorage(
        table_name="saathi_sessions",
        db_file=MEMORY_DB,
    )
    return memory, storage


# ──────────────────────────────────────────────────────────────────────────────
#  SHARED CONVERSATION LOOP
#  Both agents call this after building their Agent instance.
# ──────────────────────────────────────────────────────────────────────────────

def _conversation_loop(agent, mode_label: str) -> None:
    voice_mode = VOICE_READY
    _print_banner(voice_mode, mode_label)

    if not VOICE_READY and "LFM" in mode_label:
        print(
            "ℹ️  Offline text mode: please type in English or romanised Hinglish.\n"
            "   Hindi/Telugu script works best via voice mode (needs SARVAM_API_KEY).\n"
        )

    while True:
        try:
            user_input = _get_input(voice_mode)
            if user_input is None:
                continue
            cmd = user_input.lower().strip()
            if cmd in ("exit", "quit", "bye", "बाय"):
                break
            if cmd == "voice":
                voice_mode = not voice_mode
                print(f"  Voice: {'ON 🔊' if voice_mode else 'OFF ⌨️'}\n")
                continue

            print("💭 …")
            # stream=False → we get the full text, then pipe it to TTS
            response = agent.run(user_input, stream=False)
            text = (response.content or "").strip() or "(no response)"
            print(f"\nSaathi: {text}\n")
            if voice_mode:
                speak(text)

        except KeyboardInterrupt:
            break

    print("\n[Saathi] Goodbye!\n")


# ──────────────────────────────────────────────────────────────────────────────
#  CLOUD AGENT — Gemini 3.1 Flash-Lite
# ──────────────────────────────────────────────────────────────────────────────

def run_gemini_agent() -> None:
    from agno.agent import Agent
    from agno.models.google import Gemini

    print(f"[Saathi] Booting cloud agent ({GEMINI_MODEL}) …")
    model = Gemini(id=GEMINI_MODEL)
    memory, storage = _build_memory_and_storage(model)

    agent = Agent(
        name="Saathi",
        model=model,
        tools=SAATHI_TOOLS,
        memory=memory,
        storage=storage,
        enable_agentic_memory=True,      # extracts + stores facts per session
        add_history_to_messages=True,    # loads last N runs from DB on each turn
        num_history_runs=10,
        description=SAATHI_SYSTEM,
        markdown=False,                  # plain text plays better via TTS
        show_tool_calls=False,
    )
    _conversation_loop(agent, GEMINI_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
#  LOCAL AGENT — LFM2.5-1.2B via agno.models.llama_cpp.LlamaCpp
# ──────────────────────────────────────────────────────────────────────────────

def _find_llama_server() -> Optional[str]:
    import shutil
    for name in ["llama-server"]:
        p = shutil.which(name)
        if p:
            return p
    for p in [
        Path("/usr/local/bin/llama-server"),
        Path("/opt/homebrew/bin/llama-server"),
        Path.home() / "llama.cpp/build/bin/llama-server",
        Path.home() / ".local/bin/llama-server",
    ]:
        if p.exists():
            return str(p)
    return None


def _download_gguf() -> Optional[Path]:
    if GGUF_FILE.exists():
        print(f"✅  GGUF cached: {GGUF_FILE}")
        return GGUF_FILE
    print(f"📥  Downloading {LFM_HF_FILE} (~750 MB, first run only) …")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "huggingface_hub", "download",
                "--repo-type", "model",
                LFM_HF_REPO, LFM_HF_FILE,
                "--local-dir", str(GGUF_DIR),
                "--local-dir-use-symlinks", "False",
            ],
            check=True,
        )
        if GGUF_FILE.exists():
            print(f"✅  Downloaded: {GGUF_FILE}")
            return GGUF_FILE
    except subprocess.CalledProcessError as exc:
        print(f"❌  Download failed: {exc}")
    except FileNotFoundError:
        print("❌  huggingface_hub missing. Run: pip install huggingface_hub")
    return None


def _start_llama_server(gguf_path: Path, llama_bin: str) -> Optional[subprocess.Popen]:
    """
    Start llama-server with flags tuned for Saathi's use-case:

    --jinja     Use the model's own Jinja chat template.
                CRITICAL for LFM2.5: this activates the native
                <|tool_call_start|> / <|tool_call_end|> token handling
                so function calls are formatted exactly as the model
                was trained to produce them.

    -np 1       One parallel slot — single user, saves RAM.

    -t N        CPU threads (cores-1, max 8).

    --no-mmap   Don't memory-map weights. More stable on RAM-constrained
                edge devices; avoids page-fault latency spikes mid-sentence.

    -ngl 0      No GPU layer offload. Set to 99 if a GPU is available —
                llama.cpp will auto-detect and offload as many layers as fit.
    """
    cpu_count = os.cpu_count() or 4
    threads   = max(2, min(cpu_count - 1, 8))

    cmd = [
        llama_bin,
        "-m", str(gguf_path),
        "-c", str(LLAMA_CTX),
        "--port", str(LLAMA_PORT),
        "--host", "127.0.0.1",
        "-np", "1",
        "-t", str(threads),
        "--jinja",          # ← native LFM2.5 chat template (tool tokens!)
        "--no-mmap",
        "-ngl", "0",
        "--log-disable",
    ]
    print(f"[llama-server] port={LLAMA_PORT}  threads={threads}  ctx={LLAMA_CTX}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    base = f"http://127.0.0.1:{LLAMA_PORT}"
    for _ in range(30):        # up to 15 s
        time.sleep(0.5)
        try:
            if requests.get(f"{base}/health", timeout=1).ok:
                print(f"✅  llama-server ready at {base}")
                return proc
        except requests.ConnectionError:
            pass

    print("❌  llama-server failed to start in 15 s")
    proc.terminate()
    return None


def run_local_fallback_agent() -> None:
    llama_bin = _find_llama_server()
    if not llama_bin:
        print(
            "❌  llama-server not found.\n"
            "    macOS  :  brew install llama.cpp\n"
            "    Linux  :  download from github.com/ggml-org/llama.cpp/releases\n"
            "    Build  :  git clone … && cmake -B build && cmake --build build -j\n"
        )
        return

    gguf_path = _download_gguf()
    if not gguf_path:
        return

    server_proc = _start_llama_server(gguf_path, llama_bin)
    if not server_proc:
        return

    try:
        from agno.agent import Agent
        from agno.models.llama_cpp import LlamaCpp
    except ImportError as exc:
        print(f"❌  agno not installed: {exc}")
        server_proc.terminate()
        return

    print("[Saathi] Building local LFM2.5 agent (Agno + LlamaCpp) …")

    # LlamaCpp is Agno's OpenAILike subclass pointing at llama-server.
    # All of Agno's memory, storage, and tool-calling machinery works
    # identically to the cloud path.
    model = LlamaCpp(
        id=LFM_HF_FILE,
        base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1",
        api_key="not-needed",
        temperature=0.1,          # LFM2.5 recommended params
        top_p=0.1,
        top_k=50,
        frequency_penalty=0.05,   # ≈ repetition_penalty 1.05
        max_tokens=512,
    )
    memory, storage = _build_memory_and_storage(model)

    agent = Agent(
        name="Saathi",
        model=model,
        tools=SAATHI_TOOLS,
        memory=memory,
        storage=storage,
        enable_agentic_memory=True,   # LFM2.5 handles simple fact extraction well
        add_history_to_messages=True,
        num_history_runs=5,           # fewer than cloud to fit 8K context
        description=SAATHI_SYSTEM,
        markdown=False,
        show_tool_calls=False,
    )

    label = f"LFM2.5-1.2B Q4_K_M · llama-server :{LLAMA_PORT}"
    try:
        _conversation_loop(agent, label)
    finally:
        print("\n[Saathi] Stopping llama-server …")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


# ──────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner(voice_mode: bool, mode_label: str) -> None:
    print(f"\n{'='*58}")
    print(f"  Saathi [{mode_label}]")
    if voice_mode:
        print(f"  🔊 Voice ON  (Sarvam {SARVAM_STT_MODE} STT + Bulbul TTS)")
        print("  Press ENTER with no text to record your voice")
    else:
        print("  ⌨️  Text mode  (set SARVAM_API_KEY to enable voice)")
    print("  Type 'voice' to toggle  |  'exit' to quit")
    print(f"{'='*58}\n")


def _get_input(voice_mode: bool) -> Optional[str]:
    prompt = "🎤  ENTER=record / type: " if voice_mode else "You: "
    try:
        raw = input(prompt).strip()
    except EOFError:
        return "exit"
    if voice_mode and raw == "":
        transcript = voice_input()
        if not transcript:
            print("⚠️  Nothing captured — try again.\n")
            return None
        return transcript
    return raw or None


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import platform
    print("=" * 58)
    print("  Saathi — Voice-First WhatsApp Companion")
    print(f"  OS    : {platform.system()} {platform.release()}")
    print(f"  Voice : {'Sarvam ' + SARVAM_STT_MODE + ' + Bulbul ✅' if VOICE_READY else 'Disabled (set SARVAM_API_KEY)'}")
    print(f"  DB    : {MEMORY_DB}")
    print("=" * 58 + "\n")

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            run_gemini_agent()
        except Exception as exc:
            print(f"\n⚠️  Cloud agent failed: {exc}")
            print("    Falling back to local LFM2.5 …\n")
            run_local_fallback_agent()
    else:
        print("ℹ️  No GEMINI_API_KEY — starting local LFM2.5 fallback.\n")
        run_local_fallback_agent()


if __name__ == "__main__":
    main()