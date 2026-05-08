# Deploy on Hetzner Cloud (CX22, ~€4/mo)

End-to-end deploy of the Python agent on a Hetzner VPS, fronted by Caddy
auto-TLS, then connected to the Vercel frontend.

## 0. Pre-reqs

- Hetzner Cloud account → https://www.hetzner.com/cloud
- An SSH key on your Windows machine: `ssh-keygen -t ed25519` (skip if you have one)
- A domain or subdomain you can point at the VPS IP. Free option:
  https://www.duckdns.org → sign in with GitHub → make a token → make a
  subdomain like `ergo-agent.duckdns.org`

## 1. Create the server

Hetzner Cloud Console:

1. **+ Add server**
2. Location: **Helsinki** (lowest latency to TH from Hetzner) or Falkenstein/Nuremberg
3. Image: **Ubuntu 24.04**
4. Type: **CX22** (Shared vCPU, 2 vCPU AMD, 4 GB RAM, €3.79/mo)
5. Networking: enable IPv4 (and IPv6 if you want)
6. SSH key: paste the contents of `C:\Users\<you>\.ssh\id_ed25519.pub`
7. Cloud config (User data) — paste this to auto-install Docker on first boot:
   ```yaml
   #cloud-config
   package_update: true
   package_upgrade: true
   packages:
     - ca-certificates
     - curl
     - git
     - ufw
   runcmd:
     - install -m 0755 -d /etc/apt/keyrings
     - curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
     - chmod a+r /etc/apt/keyrings/docker.asc
     - echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" > /etc/apt/sources.list.d/docker.list
     - apt-get update
     - apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
     - ufw --force enable
     - ufw allow OpenSSH
     - ufw allow 80/tcp
     - ufw allow 443/tcp
   ```
8. **Create & buy now** — server is ready in ~30s
9. Note the **public IPv4** of the server

## 2. Point the domain at the VPS

DuckDNS: edit your subdomain → set IP to the VPS IPv4 → Update.
Wait ~1 min for propagation. Verify:
```powershell
nslookup ergo-agent.duckdns.org
```
Should return your VPS IP.

## 3. SSH in & deploy

From Windows PowerShell:

```powershell
ssh root@<VPS_IPv4>
```

On the VPS:

```bash
# 1. Clone the repo
git clone https://github.com/kongphop1100/ergo-ai-agent.git
cd ergo-ai-agent

# 2. Create the .env from the example
cp .env.example .env
nano .env
# fill in:
#   STREAM_API_KEY=...
#   STREAM_API_SECRET=...
#   OPENAI_API_KEY=...
#   OPENROUTER_API_KEY=...
# add at the bottom:
#   ERGO_DOMAIN=ergo-agent.duckdns.org
# Ctrl+O Enter Ctrl+X to save & exit.

# 3. Build & run
docker compose up -d --build

# 4. Watch logs (Ctrl+C to stop watching, containers keep running)
docker compose logs -f
```

First boot: Caddy requests TLS cert from Let's Encrypt — takes ~10s.
Then YOLO loads on first request — ~30-60s.

Verify:
```bash
curl https://ergo-agent.duckdns.org/health
# expect: HTTP/2 200
```

## 4. Wire Vercel

Go to Vercel project → **Settings → Environment Variables** → set
**`AGENT_BACKEND_URL`** = `https://ergo-agent.duckdns.org` →
**Redeploy** the latest deployment.

(Or for the first time, Vercel **Import Project** as in the README, set Root
Directory to `frontend`, and supply the three env vars including this URL.)

## 5. Update / redeploy

When you push to GitHub:
```bash
ssh root@<VPS_IPv4>
cd ergo-ai-agent
git pull
docker compose up -d --build
```

## 6. Costs

- Hetzner CX22: €3.79/mo + €0.50 IPv4 = **~€4.30 (~$4.50)/mo**
- DuckDNS subdomain: free
- Let's Encrypt cert: free (auto-renewed by Caddy)

## Troubleshooting

- `docker compose logs agent` — Python agent logs (look for `🎥 Video track
  initialized`, `🎯 rom_session processor started`, `✅ DIAG: track ADDED`)
- `docker compose logs caddy` — TLS cert acquisition log
- OOM kill: `dmesg | grep -i kill` — if seen, CX22 RAM exhausted, upgrade to
  CX32 (8 GB, €6.99/mo)
- WebRTC media: only outgoing UDP needed — Hetzner allows by default. No port
  forwarding required for media.
