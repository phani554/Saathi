# Saathi
Voice First Agent Orchestration for seniors and people with accessibility issues to bridge the digital divide and aim to be a daily lifestyle companion

# whatsapp-bridge

Bare minimum WhatsApp ↔ HTTP bridge.  
One Baileys socket, one Express server, six endpoints.

---

## What it does

```
Your Agent
    │
    │  POST /send { phone, message }          → delivers message via WA
    │  GET  /messages/:phone?limit=10         → last N messages (context)
    │  GET  /messages                         → list all active chats
    │  GET  /health                           → is socket alive?
    │  GET  /auth/qr                          → QR string to link phone
    │  POST /auth/logout                      → unlink + wipe creds
    │
    ▼
[ Express HTTP ]
    │
    ▼
[ Baileys socket ]  ←──── WhatsApp Web protocol ────► Phone
    │
    ▼
[ in-memory store ]  — Map<jid → last 50 messages>
```

---

## Quick start

```bash
npm install
npm start
```

First boot → no `auth/` dir → Baileys generates a QR.

**Option A — terminal QR** (printed automatically)  
Scan it with WhatsApp → Settings → Linked Devices → Link a Device.

**Option B — via HTTP**
```bash
curl http://localhost:3000/auth/qr
# returns { "qr": "<raw string>" }
# paste into https://qrcode.tec-it.com/ to render it
```

Once linked, the terminal prints `[socket] connected` and all endpoints become live.

---

## Endpoints

### `GET /health`
```json
{ "ok": true, "connected": true, "qrPending": false }
```
Poll this before any other call. If `connected: false` and `qrPending: true`, show the QR.

---

### `GET /auth/qr`
Returns the raw QR string while waiting for phone scan.  
- `202` — socket not ready yet, retry in 2s  
- `409` — already connected, no QR needed  
- `200` — `{ "qr": "..." }`

---

### `POST /auth/logout`
Revokes the linked device and wipes `auth/`.  
Next `npm start` will show a fresh QR.

---

### `POST /send`
```json
{ "phone": "+919876543210", "message": "Hello from the agent!" }
```
`phone` — E.164 or digits only, country code required.  
Returns `{ "ok": true, "to": "919876543210@s.whatsapp.net" }`.

---

### `GET /messages/:phone?limit=10`
```
GET /messages/919876543210?limit=10
```
Returns the last N messages (default 10, max 50) for that chat, oldest first.  
Agent uses this to build context before crafting a reply.

```json
{
  "jid": "919876543210@s.whatsapp.net",
  "messages": [
    { "id": "ABC123", "from": "919876543210@s.whatsapp.net", "to": "me", "body": "Hi!", "ts": 1712345678000, "fromMe": false },
    { "id": "DEF456", "from": "me", "to": "919876543210@s.whatsapp.net", "body": "Hey there", "ts": 1712345690000, "fromMe": true }
  ]
}
```

---

### `GET /messages`
Lists every JID that has messages in the store, with count and latest message.  
Useful for the agent to discover which conversations are active.

---

## Typical agent loop

```
1. GET /health                         → confirm connected
2. GET /messages/:phone?limit=10       → load recent context
3. agent processes context + new input
4. agent crafts reply
5. POST /send { phone, message }       → deliver it
```

---

## Notes

- **Auth persists** in `./auth/` (Baileys multi-file store). Safe to restart.
- **Messages are in-memory only** — restart clears them. Baileys will re-deliver
  unread messages on reconnect which repopulates the store.
- **Group chats** — JIDs end in `@g.us`. The bridge stores them the same way.
  Pass the full group JID (e.g. `120363XXXXXXX@g.us`) to `/send` directly.
- **No webhook** — the bridge is pull-only. If you want push (agent gets notified
  on new message), add a `WEBHOOK_URL` env var and POST to it inside the
  `messages.upsert` handler. That's the next logical extension.

---

## Reference

- [Baileys docs](https://github.com/WhiskeySockets/Baileys)
- [OpenClaw WhatsApp extension](https://github.com/openclaw/openclaw/tree/main/extensions/whatsapp) — production reference for groups, multi-account, media, pairing policy
- [OpenClaw channel docs](https://docs.openclaw.ai/channels/whatsapp)
