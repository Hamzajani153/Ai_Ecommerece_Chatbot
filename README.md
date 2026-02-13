# PC Parts Chatbot

AI-powered chatbot for PC component recommendations, pricing, and inventory management using RAG (Retrieval-Augmented Generation).

## Features

- **AI Assistant**: ChatGPT-like interface for PC parts queries
- **Product Search**: Find components with pricing in Omani Rials (OMR)
- **Multi-currency Support**: Automatic USD/EUR/GBP to OMR conversion
- **Document Processing**: Upload Excel/PDF product catalogs
- **User Authentication**: Secure JWT-based login/signup
- **Chat History**: Persistent conversation storage
- **Streaming Responses**: Real-time AI responses

## Tech Stack

**Backend:**
- FastAPI (Python)
- LangChain + OpenAI GPT-4
- ChromaDB (Vector Database)
- MongoDB (User & Chat Storage)
- PyMuPDF (PDF Processing)

**Frontend:**
- Streamlit

**Deployment:**
- Docker + Docker Compose

## Prerequisites

- Docker & Docker Compose
- OpenAI API Key
- MongoDB Atlas account (or local MongoDB)

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Hamzajani153/Ai_Ecommerece_Chatbot.git
```

### 2. Setup Environment Variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/dbname
JWT_SECRET_KEY=your-secure-random-string
OPENAI_API_KEY=sk-proj-your-openai-key
```


### 4. Build & Run
```bash
docker compose build --no-cache
docker compose up -d
```

### 5. Access Application
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health



## Load Product Data

### Via Frontend (Admin Panel):
1. Login to http://localhost:8501
2. Open sidebar → Admin Panel
3. Upload Excel/PDF file
4. Click "Load Data"


**Supported Formats:**
- Excel: `.xlsx`, `.xls`
- PDF: `.pdf`

## Common Commands

```bash
# View logs
docker compose logs -f

# Restart services
docker compose restart

# Stop everything
docker compose down

# Rebuild after code changes
docker compose down
docker compose build --no-cache
docker compose up -d

# Check running containers
docker ps
```


## API Endpoints

### Authentication
- `POST /auth/signup` - Create account
- `POST /auth/login` - Login user

### Chat
- `POST /chat` - Send message (streaming response)
- `GET /chat/history` - Get conversation history
- `DELETE /chat/history` - Clear chat history

### Admin
- `POST /admin/load-data` - Load product catalog

### Health
- `GET /health` - System status

## Production Deployment

### Cloud Platforms:
- **AWS**: Elastic Container Service (ECS) or EC2
