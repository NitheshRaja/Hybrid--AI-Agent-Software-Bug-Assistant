# 🐛 IntelliBug AI — Hybrid AI Bug Management System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_Flash-Cloud_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Gemma](https://img.shields.io/badge/Gemma_2B-Local_AI-FF6B35?style=for-the-badge&logo=google&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A conversational AI system that intelligently manages software bugs through natural language — combining a local privacy-preserving LLM and a cloud AI model for smart, real-time bug triage.**

[Features](#-features) • [Architecture](#-architecture) • [Tech Stack](#-tech-stack) • [Setup](#-setup) • [Usage](#-usage) • [Deployment](#-cloud-deployment)

</div>

---

## 📖 Overview

**IntelliBug AI** is a **Hybrid Agentic AI Bug Management System** that brings together the speed of cloud AI (Gemini Flash) and the privacy of a local LLM (Gemma-2B) to automate software bug lifecycle management through natural language conversation.

Instead of manually navigating dashboards and forms, engineering teams can simply chat with IntelliBug to:
- Create and search bug tickets
- Update ticket priorities and statuses
- Get AI-powered root cause analysis
- Query the database using plain English

The system uses an **Agentic AI approach** with dynamic tool calling — intelligently deciding when to query the database, trigger a web search, or escalate to the cloud model.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Hybrid LLM Routing** | Automatically routes queries between local Gemma-2B and cloud Gemini Flash |
| 💬 **Conversational Bug Triage** | Create, search, update, and close tickets via natural language |
| 🔧 **Agentic Tool Calling** | Dynamic tool selection for DB operations, Google Search, and GitHub |
| 🔒 **Privacy-First Mode** | Sensitive queries processed entirely on-device via Gemma-2B |
| 🗄️ **PostgreSQL Integration** | Full-featured bug ticket database with relational queries |
| 🌐 **Google Search Integration** | Real-time root cause analysis using live web search |
| 📊 **Web Chat UI** | Clean, responsive chat interface built with Flask |
| 🐳 **Docker Ready** | One-command containerized deployment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IntelliBug AI                            │
│                                                                 │
│   User ──▶  Flask Web UI  ──▶  Hybrid Agent Router             │
│                                        │                        │
│                          ┌─────────────┴──────────────┐        │
│                          ▼                            ▼         │
│                ┌──────────────────┐      ┌─────────────────┐   │
│                │   Local LLM      │      │   Cloud LLM     │   │
│                │  (Gemma-2B)      │      │ (Gemini Flash)  │   │
│                │  Privacy-First   │      │  High-Accuracy  │   │
│                └──────────────────┘      └─────────────────┘   │
│                          │                            │         │
│                          └──────────┬─────────────────┘        │
│                                     ▼                           │
│                      ┌──────────────────────────┐              │
│                      │      Agentic Tools        │              │
│                      │  ┌──────────┐ ┌────────┐ │              │
│                      │  │PostgreSQL│ │Google  │ │              │
│                      │  │   DB     │ │Search  │ │              │
│                      │  └──────────┘ └────────┘ │              │
│                      └──────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, Flask |
| **Local LLM** | Gemma-2B (via llama.cpp / Ollama) |
| **Cloud LLM** | Google Gemini Flash |
| **AI Framework** | Google Agent Development Kit (ADK) |
| **Database** | PostgreSQL 16 |
| **External Tools** | Google Search API, GitHub MCP Server |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Docker, Google Cloud Run |

---

## 📋 Prerequisites

Before running IntelliBug AI, ensure you have the following installed:

- **Python** 3.11 or higher
- **PostgreSQL** 16+
- **Ollama** — for running Gemma-2B locally ([Install Ollama](https://ollama.ai/))
- **Google API Key** — for Gemini Flash ([Get API Key](https://aistudio.google.com/app/apikey))
- **Docker** *(optional)* — for containerized deployment

---

## ⚡ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NitheshRaja/Hybrid--AI-Agent-Software-Bug-Assistant.git
cd Hybrid-AI-Agent-Software-Bug-Assistant
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Using `uv` (faster alternative):**
> ```bash
> pip install uv
> uv sync
> ```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
# Google AI (Gemini Flash)
GOOGLE_API_KEY=your_gemini_api_key_here

# PostgreSQL Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tickets_db
DB_USER=postgres
DB_PASSWORD=your_password_here

# Optional: GitHub Integration
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
```

### 4. Set Up the Database

```bash
# Connect to PostgreSQL and run the setup script
psql -U postgres -f setup_database.sql
```

This creates the `tickets` table and seeds it with sample bug data.

### 5. Download the Local Model (Gemma-2B)

```bash
# Pull Gemma-2B via Ollama
ollama pull gemma:2b

# Or use the provided download script
python download_gemma.py
```

### 6. Run the Application

```bash
python web_ui.py
```

Open your browser at **`http://localhost:7860`** and start chatting!

---

## 🚀 Docker Deployment

```bash
# Build the Docker image
docker build -t intellibug-ai .

# Run the container
docker run -p 7860:7860 --env-file .env intellibug-ai
```

---

## 💬 Usage

Interact with IntelliBug AI through the chat interface using natural language:

### 🔍 Search & Query Bugs
```
"Show me all open P0 critical bugs"
"Are there any issues assigned to samuel.green@example.com?"
"How many bugs are currently in progress?"
```

### 📝 Create New Tickets
```
"Create a new bug: Login page crashes on Safari iOS 17, assign to the frontend team, priority P1"
"File a ticket for the payment gateway timeout — it's a P0 issue"
```

### ✏️ Update Existing Tickets
```
"Mark ticket #12 as resolved"
"Change the priority of the dashboard sales widget bug to P2"
"Reassign the XZ Utils CVE ticket to the security team"
```

### 🔎 AI-Powered Root Cause Analysis
```
"What are possible root causes for the login page freezing after failed attempts?"
"Search for known fixes for database connection timeouts under peak load"
```

---

## 📁 Project Structure

```
Hybrid-AI-Agent-Software-Bug-Assistant/
│
├── software_bug_assistant/         # Core agent package
│   ├── agent.py                    # Agent orchestration logic
│   ├── tools/
│   │   └── tools.py                # Tool definitions (DB, Search, GitHub)
│   └── prompts.py                  # System prompt templates
│
├── web_ui.py                       # Flask web interface & chat UI
├── run_hybrid_agent.py             # Hybrid LLM routing logic
│
├── setup_database.sql              # DB schema, tables & seed data
├── embed_tickets.py                # Vector embedding generation
├── download_gemma.py               # Local model downloader
│
├── deployment/                     # Cloud deployment configs
│   ├── mcp-toolbox/tools.yaml      # MCP Toolbox configuration
│   └── images/                     # Architecture diagrams
│
├── Dockerfile                      # Container configuration
├── pyproject.toml                  # Project metadata & dependencies
├── .env.example                    # Environment variable template
└── README.md
```

---

## 🌩️ Cloud Deployment (Google Cloud Run)

IntelliBug AI supports full deployment to **Google Cloud Run** with **Cloud SQL (PostgreSQL)**.

### Quick Deploy

```bash
# Authenticate with Google Cloud
gcloud auth login
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Build and push the container
gcloud builds submit \
  --region=us-central1 \
  --tag us-central1-docker.pkg.dev/$PROJECT_ID/intellibug/app:latest

# Deploy to Cloud Run
gcloud run deploy intellibug-ai \
  --image=us-central1-docker.pkg.dev/$PROJECT_ID/intellibug/app:latest \
  --region=us-central1 \
  --allow-unauthenticated \
  --set-env-vars=GOOGLE_API_KEY=$GOOGLE_API_KEY
```

For a full step-by-step guide including Cloud SQL setup and MCP Toolbox deployment, see the [`deployment/`](deployment/) directory.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Nithesh S** — AI & Software Developer

[![GitHub](https://img.shields.io/badge/GitHub-NitheshRaja-181717?style=flat&logo=github)](https://github.com/NitheshRaja)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-nithesh--s-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/nithesh-s-756880251)
[![Email](https://img.shields.io/badge/Email-nitheshraja46@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:nitheshraja46@gmail.com)

---

<div align="center">
  <sub>Built with ❤️ using Python · Gemma-2B · Gemini Flash · PostgreSQL</sub>
</div>
