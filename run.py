#!/usr/bin/env python3
"""
Saathi — launcher
Boots the Node.js WhatsApp bridge then the Python agent.
Both processes share stdout so all logs appear in one terminal.
"""

import os
import sys
import time
import subprocess
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    print("=" * 52)
    print("  Saathi — Voice-First WhatsApp Companion")
    print("  Starting full stack …")
    print("=" * 52 + "\n")

    base_dir      = os.path.dirname(os.path.abspath(__file__))
    bridge_dir    = os.path.join(base_dir, "whatsapp_bridge")
    bridge_script = os.path.join(bridge_dir, "src", "server.js")
    agent_script  = os.path.join(base_dir, "agent", "agent.py")

    # ── Sanity checks ────────────────────────────────────────────────────────
    if not os.path.exists(bridge_script):
        sys.exit(f"[ERROR] Bridge script not found: {bridge_script}")
    if not os.path.exists(agent_script):
        sys.exit(f"[ERROR] Agent script not found: {agent_script}")

    bridge_proc = None
    agent_proc  = None

    try:
        # 1. Start the Node.js bridge (background)
        print("[System] Starting Node.js WhatsApp bridge …")
        bridge_proc = subprocess.Popen(
            ["node", bridge_script],
            cwd=bridge_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Give Express time to bind
        time.sleep(3)

        # Quick health check — warn but don't abort if bridge isn't ready yet
        try:
            import requests
            h = requests.get("http://localhost:3000/health", timeout=2).json()
            if h.get("connected"):
                print("[System] Bridge connected ✅")
            elif h.get("qrPending"):
                print("[System] Bridge waiting for QR scan …")
            else:
                print("[System] Bridge starting up …")
        except Exception:
            print("[System] Bridge not responding yet — agent will retry.")

        print("\n[System] Starting Saathi agent …\n")

        # 2. Start the Python agent (foreground — this is where the user interacts)
        agent_proc = subprocess.Popen(
            [sys.executable, agent_script],
            cwd=os.path.join(base_dir, "agent"),
            # inherit stdin/stdout/stderr so the user can type / hear
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        agent_proc.wait()   # block until agent exits

    except KeyboardInterrupt:
        print("\n\n[System] Shutdown signal received …")

    finally:
        if agent_proc and agent_proc.poll() is None:
            agent_proc.terminate()
            print("[System] Agent stopped.")
        if bridge_proc and bridge_proc.poll() is None:
            bridge_proc.terminate()
            print("[System] Bridge stopped.")
        print("[System] Shutdown complete.\n")


if __name__ == "__main__":
    main()