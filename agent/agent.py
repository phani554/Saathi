#!/usr/bin/env python3
import os
import sys
import platform
import requests
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Automatically load the .env file for API keys
load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
const_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
BRIDGE_URL = "http://localhost:3000"
VCARD_PATH = os.path.join(const_DIR, "contacts.vcf")
NICKNAMES_PATH = os.path.join(const_DIR, "nicknames.json") # Persistent memory for aliases

# ─── NICKNAME SYSTEM ──────────────────────────────────────────────────────────
def load_nicknames() -> dict:
    if os.path.exists(NICKNAMES_PATH):
        with open(NICKNAMES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_nickname(nickname: str, real_name: str) -> str:
    """Saves a nickname mapping so the agent remembers it permanently."""
    nicks = load_nicknames()
    nicks[nickname.lower()] = real_name
    with open(NICKNAMES_PATH, "w") as f:
        json.dump(nicks, f)
    return f"Success: Whenever the user says '{nickname}', I will now search for '{real_name}'."

# ─── SHARED PYTHON TOOLS ──────────────────────────────────────────────────────
def get_contact_number(name: str) -> str:
    """Reads contacts.vcf, handles duplicate names, resolves nicknames, and filters landlines."""
    try:
        # 1. Resolve Nickname first
        nicks = load_nicknames()
        search_name = nicks.get(name.lower(), name)

        if not os.path.exists(VCARD_PATH): return "Error: contacts.vcf not found."
        
        with open(VCARD_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            vcard_data = f.read()
            
        vcards = vcard_data.split("BEGIN:VCARD")
        matches = {}
        
        for card in vcards:
            if not card.strip(): continue
            
            fn_match = re.search(r'^FN:(.+)$', card, re.MULTILINE)
            if not fn_match: continue
            
            full_name = fn_match.group(1).strip()
            
            if search_name.lower() in full_name.lower():
                tel_matches = re.findall(r'^TEL.*?:(.+)$', card, re.MULTILINE)
                mobile_number = None
                
                for raw_number in tel_matches:
                    clean_number = re.sub(r'[^\d+]', '', raw_number)
                    
                    if clean_number.startswith("0") or clean_number.startswith("+910"):
                        continue
                    
                    digits_only = re.sub(r'\D', '', clean_number)
                    if len(digits_only) >= 10:
                        mobile_number = clean_number
                        break 
                
                if mobile_number:
                    matches[full_name] = mobile_number
                    
        if len(matches) == 0:
            if search_name != name:
                return f"Error: Contact '{search_name}' (mapped from nickname '{name}') not found."
            return f"Error: Contact '{name}' not found."
        elif len(matches) == 1:
            return list(matches.values())[0]
        else:
            names_found = ", ".join(matches.keys())
            return f"Clarification Needed: I found multiple contacts for '{search_name}': {names_found}. Ask the user which one they meant."
            
    except Exception as e: 
        return f"Error: {str(e)}"

def get_group_id(group_name: str) -> str:
    try:
        res = requests.get(f"{BRIDGE_URL}/groups")
        if res.status_code == 200:
            for g in res.json().get("groups", []):
                if group_name.lower() in g["name"].lower(): return f"{g['id']}"
        return "Group not found."
    except: return "Error fetching groups from bridge."

def send_whatsapp_message(identifier: str, message: str) -> str:
    try:
        res = requests.post(f"{BRIDGE_URL}/send", json={"phone": identifier, "message": message})
        if res.status_code == 200: return "Message sent successfully."
        return f"Failed to send: {res.text}"
    except: return "Bridge connection error."

def manage_whatsapp_session(action: str) -> str:
    try:
        if action == "logout":
            requests.post(f"{BRIDGE_URL}/auth/logout")
            return "Logged out successfully."
        elif action == "start":
            requests.post(f"{BRIDGE_URL}/start")
            return "Booting new session. Fetch QR in 3 seconds."
        elif action == "get_qr":
            res = requests.get(f"{BRIDGE_URL}/auth/qr")
            return res.json().get('qr') if res.status_code == 200 else "QR not ready."
        return "Invalid action provided."
    except: return "Bridge connection error."

AVAILABLE_TOOLS = {
    "get_contact_number": get_contact_number,
    "get_group_id": get_group_id,
    "send_whatsapp_message": send_whatsapp_message,
    "manage_whatsapp_session": manage_whatsapp_session,
    "save_nickname": save_nickname
}

# ─── AGNO GEMINI (PRIMARY CLOUD AGENT) ────────────────────────────────────────
def run_gemini_agent():
    from agno.agent import Agent
    from agno.models.google import Gemini

    print("[system] Booting Primary Cloud Agent (Gemini 3.1 Flash-Lite)...")
    
    agent = Agent(
        name="Saathi",
        model=Gemini(id="gemini-3.1-flash-lite-preview"), 
        tools=[get_contact_number, get_group_id, send_whatsapp_message, manage_whatsapp_session, save_nickname],
        instructions="""
        You are 'Saathi', a warm, patient, and reliable daily lifestyle assistant designed for seniors. 
        Tone: Friendly, respectful, clear, and concise.

        LANGUAGE & MEMORY RULES:
        1. MULTILINGUAL: You natively understand over 50 languages. Always reply in the user's language.
        2. CONTEXT: You have conversational memory. Remember previous turns and confirmations.
        
        NICKNAME RULE (CRITICAL):
        If a user asks to message someone using a nickname or relationship (e.g., "my boss", "mom", "doctor") and the `get_contact_number` tool fails to find them, ask the user for their real saved name. 
        ONCE THE USER TELLS YOU THE REAL NAME, you MUST immediately use the `save_nickname` tool to permanently map it (e.g., ARG1="boss", ARG2="Rahul Sharma") before proceeding.

        SOP - SENDING MESSAGES (STRICT COMPLIANCE REQUIRED):
        Step 1: GATHER. Find exact message and recipient. 
        Step 2: LOOKUP. Use 'get_contact_number' or 'get_group_id'.
        Step 3: DRAFT. Tell the user the draft message. YOU MUST EXPLICITLY ASK: "Should I send this?"
        Step 4: EXECUTE. ONLY AFTER the user confirms with a "yes", execute 'send_whatsapp_message'.
        Step 5: CONFIRM. Say "Message sent successfully."
        """
    )

    print("\n[Saathi-Cloud] Ready! Type 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            agent.print_response(user_input, stream=True)
            print("\n")
        except KeyboardInterrupt: break


# ─── LOCAL LFM FALLBACK (BARE-METAL) ──────────────────────────────────────────
def tune_cpu():
    cores = os.cpu_count() or 4
    torch.set_num_threads(max(1, min(4, cores // 2)))
    try: torch.set_num_interop_threads(1)
    except: pass

def load_tokenizer_model(local_only: bool):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.float32,
        local_files_only=local_only, low_cpu_mem_usage=True
    )
    model.eval()
    return tok, model

def parse_tool_command(response: str):
    match = re.search(r'\[TOOL:\s*(\w+),\s*ARG1:\s*(.*?)(?:,\s*ARG2:\s*(.*?))?\]', response)
    if match:
        tool_name = match.group(1)
        arg1 = match.group(2).strip('"\' ') if match.group(2) else None
        arg2 = match.group(3).strip('"\' ') if match.group(3) else None
        return tool_name, arg1, arg2
    return None, None, None

def run_local_fallback_agent():
    print("[system] Network/API Error detected. Booting Local Fallback (Liquid 1.2B Instruct)...")
    tune_cpu()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        tok, model = load_tokenizer_model(local_only=True)
        print("✅ Local Model loaded from cache.")
    except Exception:
        print("❌ Model not cached. Connect to internet once to download.")
        return

    print("\n[Saathi-Local] Ready! Type 'exit' to stop.\n")

    system_prompt = """You are Saathi. You are currently operating in OFFLINE FALLBACK MODE.
You MUST extract actual values from the user's prompt. NEVER use literal placeholders like 'contact_name'.

You MUST follow these exact steps when sending a message:
STEP 1: Check if you know the exact phone number. If not, output exactly: [TOOL: get_contact_number, ARG1: Actual Name]
STEP 2: Wait for the System Note to return the real digits.
STEP 3: Draft the message and ask the user "Should I send this to [Name]?" 
STEP 4: Wait for user to say yes.
STEP 5: If yes, output exactly: [TOOL: send_whatsapp_message, ARG1: Digits from Step 2, ARG2: message]

If the user teaches you a nickname, use: [TOOL: save_nickname, ARG1: nickname, ARG2: real_name]

Available tools:
[TOOL: get_contact_number, ARG1: name]
[TOOL: get_group_id, ARG1: name]
[TOOL: send_whatsapp_message, ARG1: id, ARG2: text]
[TOOL: manage_whatsapp_session, ARG1: action]
[TOOL: save_nickname, ARG1: nickname, ARG2: real_name]

Keep chat replies to 1 sentence. Do NOT skip steps."""

    conversation = [{"role": "system", "content": system_prompt}]
    
    # 1. TRANSPARENCY: Announce fallback mode explicitly to the user
    fallback_greeting = "⚠️ *I am currently running in Offline Fallback Mode. My memory is shorter, but I can still securely read your contacts, learn nicknames, and send WhatsApp messages. How can I help you?*"
    print(f"Saathi: {fallback_greeting}")
    conversation.append({"role": "assistant", "content": fallback_greeting})

    # 2. MEMORY: Keep System Prompt + last 10 messages to prevent RAM crashes
    MAX_HISTORY = 11 

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            conversation.append({"role": "user", "content": user_input})
            
            # Slide window to prevent Context Overflow Memory Crashes
            if len(conversation) > MAX_HISTORY:
                conversation = [conversation[0]] + conversation[-(MAX_HISTORY-1):]
            
            enc = tok.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask", None),
                    max_new_tokens=150, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
                )
            response = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            
            print(f"Saathi: {response}")
            conversation.append({"role": "assistant", "content": response})

            if "[TOOL:" in response:
                tool_name, arg1, arg2 = parse_tool_command(response)
                
                if tool_name and tool_name in AVAILABLE_TOOLS:
                    print(f"\n⚙️ Executing {tool_name}...")
                    try:
                        if arg1 in ["contact_name", "phone_number", "id", "name"]:
                            raise ValueError(f"Model hallucinaton detected. You passed the literal word '{arg1}' instead of the actual data.")

                        if arg2: result = AVAILABLE_TOOLS[tool_name](arg1, arg2)
                        elif arg1: result = AVAILABLE_TOOLS[tool_name](arg1)
                        else: result = AVAILABLE_TOOLS[tool_name]()
                        
                        print(f"⚙️ Result: {result}\n")
                        
                        conversation.append({"role": "user", "content": f"System Note: Tool returned -> {result}. Continue."})
                        
                        # Apply sliding window before tool follow-up
                        if len(conversation) > MAX_HISTORY:
                            conversation = [conversation[0]] + conversation[-(MAX_HISTORY-1):]
                        
                        enc = tok.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                        with torch.inference_mode():
                            out = model.generate(
                                input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask", None),
                                max_new_tokens=100, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
                            )
                        follow_up = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                        print(f"Saathi: {follow_up}")
                        conversation.append({"role": "assistant", "content": follow_up})
                        
                    except Exception as e:
                        error_msg = f"Tool execution failed: {e}. You MUST pass actual names or numbers, not placeholders."
                        print(f"⚠️ {error_msg}")
                        
                        conversation.append({"role": "user", "content": f"System Note: {error_msg}"})
                        
                        # Apply sliding window before error retry
                        if len(conversation) > MAX_HISTORY:
                            conversation = [conversation[0]] + conversation[-(MAX_HISTORY-1):]
                        
                        enc = tok.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                        with torch.inference_mode():
                            out = model.generate(
                                input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask", None),
                                max_new_tokens=100, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
                            )
                        follow_up = tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                        print(f"Saathi: {follow_up}")
                        conversation.append({"role": "assistant", "content": follow_up})
                else:
                    print(f"⚠️ Could not parse tool or invalid tool requested.")
                    
        except KeyboardInterrupt: 
            break

# ─── ROUTER ───────────────────────────────────────────────────────────────────
def main():
    print("==================================================")
    print(f" OS: {platform.system()} {platform.release()}")
    print("==================================================\n")

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        try:
            run_gemini_agent()
        except Exception as e:
            print(f"⚠️ Gemini Agent failed to start: {e}")
            run_local_fallback_agent()
    else:
        print("⚠️ No GEMINI_API_KEY found in environment variables.")
        run_local_fallback_agent()

if __name__ == "__main__":
    main()