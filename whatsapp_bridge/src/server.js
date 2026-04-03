/**
 * WhatsApp Bridge Server
 * ──────────────────────
 * One Baileys socket + one Express app.
 * The socket owns the WhatsApp Web session; Express exposes it over HTTP
 * so an external agent can send/read messages without touching Baileys directly.
 *
 * Flow:
 *   Phone ──WA Web──► Baileys socket ──► in-memory store
 *                                            ▲         │
 *                                      GET /messages   │
 *                                      POST /send      │
 *                                      GET /health     │
 *                                      GET /auth/qr    │
 *                                      POST /auth/logout
 */

import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  jidNormalizedUser,
} from "@whiskeysockets/baileys";
import { Boom } from "@hapi/boom";
import express from "express";
import qrcode from "qrcode-terminal";
import pino from "pino";
import { promises as fs } from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ─── paths ───────────────────────────────────────────────────────────────────
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const AUTH_DIR = path.join(__dirname, "../auth"); // Baileys creds live here
const PORT = process.env.PORT || 3000;

// ─── logger (silent keeps Baileys noise out of your console) ─────────────────
const logger = pino({ level: "silent" });

// ─── in-memory message store ─────────────────────────────────────────────────
// Map<jid → Message[]>  — only keeps the last MAX_PER_JID messages per chat.
// Nothing is persisted to disk; restart = fresh slate.
const MAX_PER_JID = 50; // enough context; tweak as needed
const messageStore = new Map(); // jid → [{ id, from, to, body, ts, fromMe }]

function storeMessage(jid, msg) {
  // append then trim — oldest messages fall off the back
  if (!messageStore.has(jid)) messageStore.set(jid, []);
  const bucket = messageStore.get(jid);
  bucket.push(msg);
  if (bucket.length > MAX_PER_JID) bucket.shift();
}

// ─── connection state ─────────────────────────────────────────────────────────
// Shared across the socket + Express handlers so every endpoint can read it.
let sock = null; // the live Baileys socket
let qrString = null; // current QR (null once linked)
let isConnected = false;

// ─── socket factory ───────────────────────────────────────────────────────────
// Called once at boot, and again on non-logout disconnects (reconnect loop).
async function startSocket() {
  // Clear old listeners on reconnect to prevent memory leaks
  if (sock) {
    sock.ev.removeAllListeners();
  }
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  sock = makeWASocket({
    version,
    auth: state,
    logger,
    printQRInTerminal: false, // we expose QR via HTTP instead
    // browser fingerprint shown in WA Linked Devices
    browser: ["WhatsApp Bridge", "Chrome", "1.0.0"],
  });

  // ── creds changed → persist immediately ──────────────────────────────────
  sock.ev.on("creds.update", saveCreds);

  // ── connection lifecycle ──────────────────────────────────────────────────
  sock.ev.on("connection.update", ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      // new QR arrived → cache it so GET /auth/qr can serve it
      qrString = qr;
      qrcode.generate(qr, { small: true }); // also print to console for convenience
      console.log("[auth] QR ready — scan it or hit GET /auth/qr");
    }

    if (connection === "open") {
      // phone linked and socket handshake complete
      isConnected = true;
      qrString = null; // QR no longer needed
      console.log("[socket] connected");
    }

    if (connection === "close") {
      isConnected = false;
      const code = new Boom(lastDisconnect?.error)?.output?.statusCode;
      const loggedOut = code === DisconnectReason.loggedOut;

      if (loggedOut) {
        // user logged out from phone → clear auth so next boot shows a fresh QR
        console.log("[socket] logged out — delete auth/ to re-pair");
      } else {
        // any other close (network blip, WA kicked us, etc.) → reconnect
        console.log(`[socket] closed (code ${code}) — reconnecting…`);
        startSocket();
      }
    }
  });

  // ── inbound messages → store ──────────────────────────────────────────────
  // Baileys fires 'messages.upsert' for every new message batch.
  sock.ev.on("messages.upsert", ({ messages, type }) => {
    if (type !== "notify") return; // "append" = historical load, skip it

    for (const raw of messages) {
      const jid = raw.key.remoteJid;

      // ignore WA's internal pseudo-chats
      if (!jid || jid === "status@broadcast") continue;

      // extract text — covers plain text, extended text, and image captions
      const body =
        raw.message?.conversation ||
        raw.message?.extendedTextMessage?.text ||
        raw.message?.imageMessage?.caption ||
        "";

      const normalized = jidNormalizedUser(jid); // strip device suffix

      storeMessage(normalized, {
        id: raw.key.id,
        from: raw.key.fromMe ? "me" : jid,
        to: raw.key.fromMe ? jid : "me",
        body,
        ts: (raw.messageTimestamp ?? 0) * 1000, // → ms epoch
        fromMe: raw.key.fromMe ?? false,
      });
    }
  });
}

// ─── Express app ──────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());

// helper: normalise a phone number to a WA JID
// "+919876543210" or "919876543210" → "919876543210@s.whatsapp.net"
function toJid(phone) {
  // 1. If it is already a valid WhatsApp JID, return it as-is
  if (phone.includes("@g.us") || phone.includes("@s.whatsapp.net")) {
    return phone;
  }

  // 2. Strip all non-digit characters (removes +, -, spaces, etc.)
  // WhatsApp IDs require pure digits, so stripping the '+' is mandatory.
  let digits = phone.replace(/\D/g, "");

  // 3. SMART INDIAN NUMBER FORMATTING:
  if (digits.length === 10) {
    // Missing country code entirely -> add 91
    digits = "91" + digits;
  } else if (digits.length === 11 && digits.startsWith("0")) {
    // Saved with a leading 0 (e.g. 09876543210) -> replace 0 with 91
    digits = "91" + digits.substring(1);
  }

  // If the number already had a country code (like 919876543210 or 14155552671),
  // it safely bypasses the checks above and remains perfectly intact.

  // 4. Return the fully formatted JID
  return `${digits}@s.whatsapp.net`;
}

// ── GET /health ───────────────────────────────────────────────────────────────
// "is the server alive and is the socket linked?" — agent polls this first
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    connected: isConnected,
    qrPending: !!qrString, // true if waiting for phone scan
  });
});

// ── GET /auth/qr ──────────────────────────────────────────────────────────────
// Returns the raw QR string; render it however the agent/UI likes.
// If already linked → 409 so the caller knows not to show a QR.
app.get("/auth/qr", (_req, res) => {
  if (isConnected) return res.status(409).json({ error: "already connected" });
  if (!qrString)
    return res.status(202).json({ message: "QR not ready yet, retry in 2s" });

  res.json({ qr: qrString });
});

// ── POST /auth/logout ─────────────────────────────────────────────────────────
// Disconnects the socket and wipes auth/ so next boot starts fresh.
app.post("/auth/logout", async (_req, res) => {
  try {
    if (sock) {
      await sock.logout();
      sock = null;
    }
    await fs.rm(AUTH_DIR, { recursive: true, force: true });
    isConnected = false;
    qrString = null;
    res.json({ ok: true, message: "Logged out. Session wiped." });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── POST /start ──────────────────────────────────────────────────────────────
// Allows the agent to manually boot up a new Baileys session.
app.post("/start", async (_req, res) => {
  if (isConnected || qrString) {
    return res
      .status(400)
      .json({ error: "A session is already active or pending." });
  }

  try {
    startSocket();
    res.json({
      ok: true,
      message: "Booting new session. Fetch the QR code in a few seconds.",
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── POST /send ────────────────────────────────────────────────────────────────
// Agent posts here with { phone, message } to deliver a WhatsApp message.
// Body:  { "phone": "+919876543210", "message": "Hello!" }
// This is the main endpoint the agent calls after crafting its reply.
app.post("/send", async (req, res) => {
  if (!isConnected) return res.status(503).json({ error: "not connected" });

  try {
    const { phone, message } = req.body;
    if (!phone || !message) {
      return res.status(400).json({ error: "phone and message are required" });
    }

    const jid = toJid(phone);

    // FIX: Add this safety check! Prevent sending to empty/invalid JIDs
    if (jid === "@s.whatsapp.net" || jid.length < 15) {
      return res
        .status(400)
        .json({ error: `Invalid phone number format: ${phone}` });
    }

    const sentMsg = await sock.sendMessage(jid, { text: message });
    res.json({ success: true, messageId: sentMsg.key.id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── GET /messages/:phone ──────────────────────────────────────────────────────
// Returns up to 10 recent messages for a chat — agent uses this for context
// before crafting a reply ("what did we last talk about?").
// ?limit=N overrides the default 10 (max 50 — what the store holds).
app.get("/messages/:phone", (req, res) => {
  const jid = toJid(req.params.phone);
  const limit = Math.min(parseInt(req.query.limit ?? "10", 10), MAX_PER_JID);
  const msgs = messageStore.get(jid) ?? [];

  // slice from the end → most recent N messages, oldest first
  res.json({ jid, messages: msgs.slice(-limit) });
});

// ── GET /messages ─────────────────────────────────────────────────────────────
// List all JIDs that have messages in the store — lets the agent discover
// active conversations without knowing phone numbers in advance.
app.get("/messages", (_req, res) => {
  const chats = [];
  for (const [jid, msgs] of messageStore) {
    chats.push({ jid, count: msgs.length, latest: msgs.at(-1) ?? null });
  }
  res.json({ chats });
});

// ── GET /groups ──────────────────────────────────────────────────────────────
// Allows the agent to fetch all groups the phone is a part of.
// The agent can use this to map a group name (subject) to its JID.
app.get("/groups", async (_req, res) => {
  if (!isConnected) return res.status(503).json({ error: "not connected" });

  try {
    // Fetches metadata for all groups the linked number is currently in
    const groups = await sock.groupFetchAllParticipating();

    // Format it into a clean array for the agent: [{ id, name }]
    const groupList = Object.values(groups).map((g) => ({
      id: g.id,
      name: g.subject,
    }));

    res.json({ groups: groupList });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── boot ─────────────────────────────────────────────────────────────────────
// Start the Baileys socket first, then open the HTTP server.
// If auth/ already has valid creds, the socket will connect without a QR.
startSocket()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`[server] listening on http://localhost:${PORT}`);
      console.log(`[server] endpoints:`);
      console.log(`         GET  /health`);
      console.log(`         GET  /auth/qr`);
      console.log(`         POST /auth/logout`);
      console.log(`         POST /send          { phone, message }`);
      console.log(`         GET  /messages/:phone?limit=10`);
      console.log(`         GET  /messages       (list all chats)`);
      console.log(`         GET  /groups         (list all groups)`);
    });
  })
  .catch(console.error);
