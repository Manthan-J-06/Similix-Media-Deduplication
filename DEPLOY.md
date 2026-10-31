# Argus — Production Deployment Guide

This guide covers deployment instructions for three target environments:
1. **Local Docker / Docker Compose**
2. **Cloudflare Tunnel from a Mac**
3. **Hugging Face Space (Docker SDK)**

---

## 1. Local Docker Deployment (docker-compose)

Argus is containerized using `python:3.11-slim`, runs as a non-root user (`argus`, UID 1000), uses CPU-only PyTorch to minimize image footprint, and pre-downloads open_clip model weights during the build so container startup is fast and offline-capable.

### Prerequisites
- Docker Engine 24.0+ and Docker Compose v2+ installed.

### Setup & Launch

1. **Configure Environment Variables (Optional)**
   Create a `.env` file in the repository root or export variables in your shell:
   ```bash
   SECRET_KEY=change-this-to-a-secure-random-string
   ALLOW_REGISTRATION=true
   HOSTED_MODE=false
   BASE_URL=http://localhost:8000
   ```
   > [!NOTE]
   > Never bake secrets or private credentials into the Dockerfile or commit `.env` into git.

2. **Build and Run with Docker Compose**
   ```bash
   docker compose up --build -d
   ```

3. **Verify Service Health & Startup Self-Check**
   ```bash
   curl http://localhost:8000/healthz
   # Response: {"status":"ok","service":"argus"}
   ```

   Argus logs a startup self-check banner upon boot:
   ```text
   ==================================================
     ARGUS Startup Self-Check
   ==================================================
     Data Directory : /data
     DB Path        : /data/argus.db
     Secret Key Set : yes
   ==================================================
   ```

4. **Access the Web Interface**
   Open `http://localhost:8000` in your web browser. First-time visitors will be redirected to `http://localhost:8000/login` to register an account.

5. **Stop the Container**
   ```bash
   docker compose down
   ```
   > [!CAUTION]
   > **Never use `docker compose down -v`**.
   > The `-v` flag deletes Docker volumes, which permanently destroys the `argus_data` volume containing `auth.db`, per-user databases, uploaded libraries, thumbnails, and ledger journals. To preserve user accounts and data across container restarts and image updates, always stop containers using `docker compose down` (without `-v`).

### What Persists vs What Doesn't (Docker)

| Data Category | Persistence | Storage Location / Explanation |
|---|---|---|
| **User Data & DBs** | ✅ **Persists** | Stored in Docker named volume `argus_data` mounted at `/data`. Retains `auth.db`, per-user `argus.db`, `library/`, `trash/`, `thumbnails/`, and `jobs/`. |
| **Model Weights** | ✅ **Persists** | Pre-cached in image layer at `/home/argus/.cache/huggingface` during image build. |
| **Container Root Filesystem** | ❌ **Ephemeral** | Any temporary logs, package installations, or scratch files outside `/data` will be reset if the container is recreated (`docker compose down && docker compose up`). |

---

## 2. Cloudflare Tunnel from this Mac

A Cloudflare Tunnel (`cloudflared`) exposes the Argus web service securely to the internet with TLS termination without opening router ports or configuring static public IP addresses.

### Prerequisites
- A domain name registered or active on Cloudflare DNS.
- Homebrew installed on macOS.

### Step-by-Step Instructions

1. **Install `cloudflared` CLI**
   ```bash
   brew install cloudflared
   ```

2. **Authenticate `cloudflared` with your Cloudflare Account**
   ```bash
   cloudflared tunnel login
   ```
   This will open your browser to select your domain and download an authentication certificate to `~/.cloudflared/cert.pem`.

3. **Create a Dedicated Tunnel**
   ```bash
   cloudflared tunnel create argus
   ```
   Note the Tunnel UUID and credentials file path printed in the output (`~/.cloudflared/<TUNNEL_ID>.json`).

4. **Configure the Tunnel**
   Create `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <TUNNEL_ID>
   credentials-file: /Users/<your-username>/.cloudflared/<TUNNEL_ID>.json

   ingress:
     - hostname: argus.yourdomain.com
       service: http://127.0.0.1:8000
     - service: http_status:404
   ```

5. **Route DNS Traffic to the Tunnel**
   ```bash
   cloudflared tunnel route dns argus argus.yourdomain.com
   ```

6. **Start Argus and the Tunnel**
   In terminal 1, start the Argus server (or run it via Docker Compose):
   ```bash
   # Run via Docker:
   BASE_URL=https://argus.yourdomain.com docker compose up -d

   # Or run directly via CLI:
   BASE_URL=https://argus.yourdomain.com argus serve --host 127.0.0.1 --port 8000
   ```

   In terminal 2, launch the tunnel:
   ```bash
   cloudflared tunnel run argus
   ```

7. **Verify Public Access**
   Navigate to `https://argus.yourdomain.com` in your browser. Cloudflare automatically provides HTTPS with zero firewall changes.

### What Persists vs What Doesn't (Cloudflare Tunnel)

| Data Category | Persistence | Storage Location / Explanation |
|---|---|---|
| **User Libraries & Ledgers** | ✅ **Persists** | Stored on the local host Mac filesystem under `~/.argus/` or inside the Docker named volume. |
| **Tunnel Connection State** | ❌ **Stateless** | Cloudflare tunnels are stateless multiplexed connections. If the Mac sleeps or connection drops, the tunnel automatically reconnects upon waking. |

---

## 3. Hugging Face Space (Docker SDK Deployment)

Hugging Face Spaces allows hosting containerized web applications directly on managed infrastructure using Docker.

### Space Configuration

1. **Create a New Space on Hugging Face**
   - Repository Type: **Space**
   - SDK: **Docker** (Blank template)
   - Visibility: Public or Private

2. **Configure Space Metadata (`README.md`)**
   In your Hugging Face Space repository, include YAML frontmatter at the top of `README.md`:
   ```yaml
   ---
   title: Argus Deduplication
   emoji: 🔍
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 7860
   ---
   ```

3. **Set Secrets and Environment Variables in Space Settings**
   Navigate to **Settings → Variables and Secrets**:
   - `PORT`: `7860` (Hugging Face sets this automatically)
   - `HOST`: `0.0.0.0`
   - `HOSTED_MODE`: `true` (Disables local directory scanning and dataset URL fetching)
   - `ALLOW_REGISTRATION`: `true` (or `false` for invite-only)
   - `SECRET_KEY`: `<minimum 32 character random string>` (Required in hosted mode)
   - `ARGUS_DATA_DIR`: `/data` (Single root for all user databases, libraries, jobs, and thumbnails)

4. **Push Repository to Hugging Face**
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/argus
   git push space main
   ```

### What Persists vs What Doesn't (Hugging Face Spaces)

> [!IMPORTANT]
> Understanding the difference between Free and Persistent Storage tiers is critical for data safety on Hugging Face Spaces.

| Tier | Path | Persistence | Explanation |
|---|---|---|---|
| **Free Tier** | `/data` or `/app` | ❌ **Ephemeral** | The container filesystem is ephemeral. Whenever the Space restarts, updates, or goes to sleep after inactivity, **all uploaded images, users, and SQLite databases will be wiped clean**. |
| **Persistent Storage Tier** | `/data` | ✅ **Persists** | When **Persistent Storage** (Small, Medium, or Large) is purchased and enabled in Space settings, `/data` is backed by network-attached storage. Setting `ARGUS_DATA_DIR=/data` guarantees that all users, credentials, image libraries, ledgers, and deduplication states **persist permanently across restarts and rebuilds**. |

---

## 4. Oracle Cloud / VPS Ubuntu Server

This section covers self-hosting Argus on any Ubuntu 22.04+ VPS (Oracle Cloud Free Tier, DigitalOcean, Linode, Hetzner, etc.) with Docker, an HTTPS reverse proxy via Caddy, and systemd for automatic restarts.

### Prerequisites

- Ubuntu 22.04+ server with a public IP address.
- A domain name with an **A record** pointing to the server's IP (for automatic HTTPS).
- SSH access with `sudo` privileges.

### Step 1: Install Docker Engine

```bash
# Update package index and install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker CE and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow your user to run Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Clone the Repository and Configure

```bash
git clone https://github.com/your-org/argus.git /opt/argus
cd /opt/argus

# Create .env with your secrets
cat > .env <<'EOF'
SECRET_KEY=<at-least-32-random-characters>
ALLOW_REGISTRATION=true
HOSTED_MODE=false
BASE_URL=https://argus.yourdomain.com
ARGUS_DATA_DIR=/data
EOF
```

> [!CAUTION]
> Generate `SECRET_KEY` with `python3 -c "import secrets; print(secrets.token_hex(32))"`. Never reuse it or commit it to version control.

### Step 3: Build and Start with Docker Compose

```bash
cd /opt/argus
docker compose up --build -d
docker compose ps        # verify argus-app is Up
docker compose logs -f   # watch startup logs
```

### Step 4: Install Caddy for Automatic HTTPS

Caddy automatically obtains and renews Let's Encrypt TLS certificates.

```bash
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy
```

Create `/etc/caddy/Caddyfile`:

```caddyfile
argus.yourdomain.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

> [!TIP]
> Caddy handles HTTP → HTTPS redirects and certificate renewal automatically. No `certbot` or manual renewal cron is needed.

### Step 5: Configure UFW Firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp    # HTTP (Caddy redirects to HTTPS)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
sudo ufw status
```

> [!IMPORTANT]
> Do **not** expose port 8000 publicly — all traffic must go through Caddy (port 443). Argus binds only on `localhost:8000`.

### Step 6: Systemd Service for Auto-Start on Boot

Create `/etc/systemd/system/argus.service`:

```ini
[Unit]
Description=Argus Deduplication Service
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/argus
ExecStart=/usr/bin/docker compose up -d --remove-orphans
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable argus
sudo systemctl start argus
```

### Step 7: Automated Backup of Data Volume

Add a daily cron job to back up the `argus_data` Docker volume:

```bash
sudo crontab -e
```

Add the following line:

```cron
0 3 * * * docker run --rm -v argus_argus_data:/data:ro -v /opt/backups/argus:/backup alpine tar czf /backup/argus_data_$(date +\%Y\%m\%d).tar.gz -C /data . && find /opt/backups/argus -name "*.tar.gz" -mtime +30 -delete
```

```bash
sudo mkdir -p /opt/backups/argus
```

> [!NOTE]
> The cron runs at 03:00 daily, creates a compressed `.tar.gz` snapshot of `/data`, and prunes backups older than 30 days.

### Step 8: Google Drive Integration (Optional)

To enable Google Drive sign-in, copy your `client_secret.json` into the Argus data volume:

```bash
docker cp /path/to/client_secret.json argus-app:/data/client_secret.json
```

Or mount it read-only via `docker-compose.yml` (add under `volumes:` in the `argus` service):

```yaml
# Uncomment and set path if client_secret.json exists on the host:
# - /path/to/client_secret.json:/data/client_secret.json:ro
```

### What Persists vs What Doesn't (Oracle/VPS)

| Data Category | Persistence | Storage Location / Explanation |
|---|---|---|
| **User Data & DBs** | ✅ **Persists** | Stored in Docker named volume `argus_data` at `/data` on the host. Survives container restarts, image rebuilds, and OS reboots (as long as the volume is not manually deleted). |
| **TLS Certificates** | ✅ **Persists** | Managed by Caddy and stored in `/var/lib/caddy/`. Auto-renewed before expiry. |
| **Daily Backups** | ✅ **Persists** | Written to `/opt/backups/argus/` on the host. Retained for 30 days. |
| **Container Layers** | ❌ **Ephemeral** | Rebuilt with `docker compose up --build`. Does not affect user data in the named volume. |
