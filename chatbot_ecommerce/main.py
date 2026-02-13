import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Import custom modules
from auth import AuthHandler, get_current_user, validate_email, validate_password
from database import (
    connect_to_mongo, close_mongo_connection, 
    UserDB, ChatHistoryDB
)
from models import (
    SignupRequest, LoginRequest, AuthResponse,
    ChatRequest, ChatHistoryResponse, ChatMessage,
    AdminLoadDataResponse, GenericResponse
)
import tiktoken
import openai
import chromadb
import re
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import fitz
import concurrent.futures
import base64
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY")

openai.api_key = OPENAI_API_KEY
GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONVERSATION_HISTORY = 10
MAX_CONTEXT_TOKENS = 8000

# Paths
DATA_FOLDER = Path("data")
CHROMA_DB_PATH = Path("data/chroma_db/product_catalog")

# Initialize FastAPI
app = FastAPI(title="PC Parts Chatbot API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
shared_db_manager: Optional['ChromaDBManager'] = None
user_message_histories: Dict[str, 'MongoMessageHistory'] = {}


# ==================== ADMIN AUTHENTICATION ====================
# No authentication required for admin endpoint


# ==================== STARTUP & SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global shared_db_manager
    
    await connect_to_mongo()
    
    # Initialize shared ChromaDB (load if exists)
    try:
        shared_db_manager = ChromaDBManager()
        doc_count = shared_db_manager.get_collection_size()
        print(f"  ChromaDB loaded with {doc_count} documents")
    except Exception as e:
        print(f" ChromaDB initialization: {str(e)}")
    
    print("  Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    await close_mongo_connection()
    print(" Server shutdown complete")


# ==================== MESSAGE HISTORY CLASS ====================
class MongoMessageHistory(BaseChatMessageHistory):
    """MongoDB-backed message history"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._messages: List[BaseMessage] = []
        self._loaded = False
    
    async def load_history(self):
        """Load chat history from MongoDB"""
        if not self._loaded:
            messages_data = await ChatHistoryDB.get_chat_history(self.user_id)
            self._messages = []
            
            for msg_data in messages_data:
                if msg_data["role"] == "user":
                    self._messages.append(HumanMessage(content=msg_data["content"]))
                elif msg_data["role"] == "assistant":
                    self._messages.append(AIMessage(content=msg_data["content"]))
            
            self._loaded = True
    
    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._trim_messages()
    
    async def save_message(self, role: str, content: str):
        """Save message to MongoDB"""
        await ChatHistoryDB.add_message(self.user_id, role, content)
    
    def clear(self) -> None:
        self._messages = []
    
    def _trim_messages(self) -> None:
        if len(self._messages) > MAX_CONVERSATION_HISTORY * 2:
            recent_messages = self._messages[-(MAX_CONVERSATION_HISTORY * 2):]
            self._messages = recent_messages


# ==================== DOCUMENT PROCESSING CLASSES ====================
class DocumentProcessor:
    
    @staticmethod
    def extract_text_with_vision(page_image_bytes: bytes) -> str:
        try:
            base64_image = base64.b64encode(page_image_bytes).decode('utf-8')
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text, describe all charts, tables, diagrams, and technical content from this image. Be comprehensive and detailed."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with GPT Vision: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_excel(file_path: Path) -> str:
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            full_text = ""
            for sheet_name, data in df.items():
                full_text += f"\n\nSHEET: {sheet_name}\n"
                full_text += data.to_string()
            return full_text
        except Exception as e:
            print(f"Error reading Excel {file_path}: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        try:
            document = fitz.open(file_path)
            full_text = ""
            
            def process_page(page_num):
                page = document.load_page(page_num)
                text = page.get_text()
                images = page.get_images(full=True)
                
                page_text = f"\n\nPAGE NO {page_num + 1}\n"
                page_text += text
                
                if images:
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        vision_content = DocumentProcessor.extract_text_with_vision(img_data)
                        if vision_content:
                            page_text += "\n--- DETAILED CONTENT FROM IMAGES ---\n"
                            page_text += vision_content + "\n"
                    except Exception as e:
                        print(f"Error processing images on page {page_num + 1}: {str(e)}")
                
                return page_num, page_text
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_page, page_num) for page_num in range(document.page_count)]
                results = concurrent.futures.as_completed(futures)
                page_results = sorted([result.result() for result in results], key=lambda x: x[0])
                for _, page_text in page_results:
                    full_text += page_text
            
            return full_text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""


class TextChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'filename': filename,
                    'token_count': current_tokens
                })
                
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(self.tokenizer.encode(current_chunk))
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunk_id = str(uuid.uuid4())
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'filename': filename,
                'token_count': current_tokens
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.tokenizer.decode(overlap_tokens)


class ChromaDBManager:
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = str(CHROMA_DB_PATH)
        
        self.persist_directory = persist_directory
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # Create directory structure
        parent_dir = os.path.dirname(persist_directory)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, mode=0o755, exist_ok=True)
        
        os.makedirs(persist_directory, mode=0o755, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection_name = "product_catalog"
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(name=self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def get_collection_size(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0
    
    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.get_embeddings(texts)
            
            if not embeddings:
                return False
            
            ids = [chunk['id'] for chunk in chunks]
            metadatas = [
                {
                    'filename': chunk['filename'],
                    'token_count': chunk['token_count']
                }
                for chunk in chunks
            ]
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {str(e)}")
            return False
    
    def clear_collection(self):
        """Clear all data from collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            print("  Collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            collection_size = self.get_collection_size()
            if collection_size == 0:
                return []
            
            actual_n_results = min(n_results, collection_size)
            query_embedding = self.get_embeddings([query])
            if not query_embedding:
                return []
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=actual_n_results
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            similar_docs = []
            for i, doc in enumerate(results['documents'][0]):
                similar_docs.append({
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
            
            return similar_docs
        except Exception as e:
            print(f"Error searching similar documents: {str(e)}")
            return []


class EnhancedRAGChatbot:
    def __init__(self, db_manager: ChromaDBManager, message_history: MongoMessageHistory):
        self.db_manager = db_manager
        self.message_history = message_history
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        self.llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=0.3,
            streaming=True,
            openai_api_key=OPENAI_API_KEY
        )

        self.system_prompt = """You are a helpful PC hardware store assistant with expertise in computer components and custom PC builds.

        Your role is to help customers find PC components, provide pricing and availability information, and answer questions about building custom PCs.

        CORE RESPONSIBILITIES:
        1. Product Information: Provide accurate details about available PC components
        2. Pricing & Stock: Always mention the sales price in Omani Rials (OMR) and available quantity
        3. Product Recommendations: Suggest compatible components and alternatives when asked
        4. Professional Tone: Be friendly, concise, and helpful

        CURRENCY HANDLING:
        - ALL prices in our database are in Omani Rials (OMR)
        - If a customer asks for products in USD, EUR, or any other currency:
        1. Convert their budget to OMR using approximate rates:
            - 1 USD ≈ 0.38 OMR
            - 1 EUR ≈ 0.42 OMR
            - 1 GBP ≈ 0.49 OMR
        2. Find products within that OMR budget
        3. Show prices in BOTH their requested currency AND OMR
        
        Example: "You asked for products under $500. That's approximately 190 OMR. Here's what we have:"

        RESPONSE GUIDELINES:
        - Keep responses SHORT and TO THE POINT
        - When a product is found: State availability, price (in OMR + requested currency if applicable), and quantity clearly
        - Format prices as: "XX OMR (≈ $YY USD)" when customer asks in other currency
        - Only recommend products that exist in the provided context

        FORMATTING FOR PC BUILD RECOMMENDATIONS:
        When recommending a complete PC build or multiple components, ALWAYS format as a structured list:

        Example format:
        **Recommended PC Build (Under $1,300 / 500 OMR):**

        **CPU:** Intel Core i9-14900K
        - Price: 320 OMR (≈ $842 USD)
        - Stock: 10 units available
        - Specs: 24 cores, 5.8GHz boost

        **GPU:** RTX 4090
        - Price: 850 OMR (≈ $2,237 USD)
        - Stock: 5 units available
        - Specs: 24GB GDDR6X, 450W TDP

        **Motherboard:** ASUS ROG Strix B760
        - Price: 84 OMR (≈ $221 USD)
        - Stock: 15 units available
        - Specs: ATX, DDR5, PCIe 5.0

        **RAM:** [Component name]
        - Price: XX OMR (≈ $YY USD)
        - Stock: X units
        - Specs: [specifications]

        **Storage:** [Component name]
        - Price: XX OMR (≈ $YY USD)
        - Stock: X units
        - Specs: [specifications]

        **Power Supply:** [Component name]
        - Price: XX OMR (≈ $YY USD)
        - Stock: X units
        - Specs: [specifications]

        **Case:** [Component name]
        - Price: XX OMR (≈ $YY USD)
        - Stock: X units
        - Specs: [specifications]

        **Total Price:** XXX OMR (≈ $YYY USD)

        IMPORTANT:
        - Each component on a new line with clear indentation
        - Always show: Component name, Price in OMR (+ converted currency if asked), Stock, Key specs
        - Add total price at the end for complete builds in BOTH currencies if customer asked in non-OMR
        - Use bold text for component categories and important info
        - When customer mentions USD/EUR/other currency, acknowledge it and show converted OMR equivalent

        Context from product database:
        {context}"""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    
    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        if not relevant_docs:
            return "No relevant documents found."
        
        context_parts = []
        total_tokens = 0
        
        for doc in relevant_docs:
            doc_context = f"Document: {doc['metadata']['filename']}\nContent: {doc['text']}"
            doc_tokens = len(self.tokenizer.encode(doc_context))
            
            if total_tokens + doc_tokens > MAX_CONTEXT_TOKENS:
                break
                
            context_parts.append(doc_context)
            total_tokens += doc_tokens
        
        return "\n\n".join(context_parts)
    
    def _get_conversation_context(self) -> List[BaseMessage]:
        messages = self.message_history.messages
        conversation_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        if len(conversation_messages) > MAX_CONVERSATION_HISTORY * 2:
            conversation_messages = conversation_messages[-(MAX_CONVERSATION_HISTORY * 2):]
        
        return conversation_messages
    
    def generate_response_stream(self, query: str, context_docs: List[Dict[str, Any]]):
        try:
            context = self._build_context(context_docs)
            chat_history = self._get_conversation_context()
            
            chain = (
                {
                    "context": lambda x: context,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: chat_history
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            for chunk in chain.stream(query):
                yield chunk
                
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def add_message_to_history(self, human_message: str, ai_message: str):
        self.message_history.add_message(HumanMessage(content=human_message))
        self.message_history.add_message(AIMessage(content=ai_message))


# ==================== HELPER FUNCTIONS ====================
async def get_user_message_history(user_id: str) -> MongoMessageHistory:
    """Get or create message history for user"""
    if user_id not in user_message_histories:
        history = MongoMessageHistory(user_id)
        await history.load_history()
        user_message_histories[user_id] = history
    return user_message_histories[user_id]


# ==================== AUTHENTICATION ENDPOINTS ====================
@app.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    """Register a new user"""
    
    if not validate_email(request.email):
        return AuthResponse(success=False, message="Invalid email format")
    
    is_valid, error_msg = validate_password(request.password)
    if not is_valid:
        return AuthResponse(success=False, message=error_msg)
    
    existing_user = await UserDB.get_user_by_email(request.email)
    if existing_user:
        return AuthResponse(success=False, message="Email already registered")
    
    hashed_password = AuthHandler.hash_password(request.password)
    user_id = await UserDB.create_user(request.email, hashed_password)
    
    if not user_id:
        return AuthResponse(success=False, message="Failed to create user")
    
    token = AuthHandler.create_access_token(user_id, request.email)
    
    return AuthResponse(
        success=True,
        message="User registered successfully",
        token=token,
        user_id=user_id,
        email=request.email
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login user"""
    
    user = await UserDB.get_user_by_email(request.email)
    if not user:
        return AuthResponse(success=False, message="Invalid email or password")
    
    if not AuthHandler.verify_password(request.password, user["password"]):
        return AuthResponse(success=False, message="Invalid email or password")
    
    token = AuthHandler.create_access_token(user["_id"], user["email"])
    
    return AuthResponse(
        success=True,
        message="Login successful",
        token=token,
        user_id=user["_id"],
        email=user["email"]
    )


# ==================== ADMIN ENDPOINT ====================
@app.post("/admin/load-data", response_model=AdminLoadDataResponse)
async def load_data_from_folder(
    clear_existing: bool = False
):
    
    global shared_db_manager
    
    if not DATA_FOLDER.exists():
        raise HTTPException(status_code=404, detail="Data folder not found")
    
    # Initialize or clear ChromaDB
    if shared_db_manager is None:
        shared_db_manager = ChromaDBManager()
    
    if clear_existing:
        shared_db_manager.clear_collection()
        print(" Cleared existing data")
    
    processor = DocumentProcessor()
    chunker = TextChunker()
    
    files_processed = 0
    total_chunks = 0
    files_loaded = []
    errors = []
    
    # Get all files in data folder
    supported_extensions = {'.xlsx', '.xls', '.pdf'}
    files = [f for f in DATA_FOLDER.iterdir() if f.suffix.lower() in supported_extensions]
    
    if not files:
        return AdminLoadDataResponse(
            success=False,
            message="No supported files found in data/ folder",
            files_processed=0,
            total_chunks=0,
            files_loaded=[],
            errors=["Supported formats: Excel (.xlsx, .xls), PDF (.pdf)"]
        )
    
    for file_path in files:
        try:
            print(f" Processing: {file_path.name}")
            
            # Extract text based on file type
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                text = processor.extract_text_from_excel(file_path)
            elif file_path.suffix.lower() == '.pdf':
                text = processor.extract_text_from_pdf(file_path)
            else:
                continue
            
            if not text.strip():
                errors.append(f"No text extracted from {file_path.name}")
                continue
            
            # Chunk and add to database
            chunks = chunker.chunk_text(text, file_path.name)
            chunk_count = len(chunks)
            
            if shared_db_manager.add_documents(chunks):
                total_chunks += chunk_count
                files_processed += 1
                files_loaded.append(file_path.name)
                print(f"  {file_path.name}: {chunk_count} chunks")
            else:
                errors.append(f"Failed to add {file_path.name} to database")
                
        except Exception as e:
            errors.append(f"Error processing {file_path.name}: {str(e)}")
            print(f" Error: {file_path.name} - {str(e)}")
    
    message = f"Successfully loaded {files_processed}/{len(files)} file(s) with {total_chunks} chunks"
    
    return AdminLoadDataResponse(
        success=files_processed > 0,
        message=message,
        files_processed=files_processed,
        total_chunks=total_chunks,
        files_loaded=files_loaded,
        errors=errors if errors else None
    )


# ==================== CHAT ENDPOINT ====================
@app.post("/chat")
async def chat_stream(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """Streaming chat endpoint - returns SSE (Server-Sent Events)"""
    
    if shared_db_manager is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    user_id = current_user["user_id"]
    
    # Check if data is loaded
    collection_size = shared_db_manager.get_collection_size()
    
    if collection_size == 0:
        async def no_docs_stream():
            response = "The product database is empty. Please contact the administrator to load product data."
            yield f"data: {json.dumps({'content': response, 'done': True})}\n\n"
        
        return StreamingResponse(
            no_docs_stream(),
            media_type="text/event-stream"
        )
    
    # Get user's message history
    message_history = await get_user_message_history(user_id)
    
    # Create chatbot instance
    chatbot = EnhancedRAGChatbot(shared_db_manager, message_history)
    
    # Search for relevant documents
    search_results = min(20, collection_size)
    relevant_docs = shared_db_manager.search_similar(request.message, search_results)
    
    if not relevant_docs:
        async def no_results_stream():
            response = "I couldn't find relevant information. Try rephrasing your query."
            yield f"data: {json.dumps({'content': response, 'done': True})}\n\n"
        
        return StreamingResponse(
            no_results_stream(),
            media_type="text/event-stream"
        )
    
    # Stream response
    async def generate_stream():
        full_response = ""
        try:
            for chunk in chatbot.generate_response_stream(request.message, relevant_docs):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                    await asyncio.sleep(0.01)
            
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            
            # Add to conversation history
            chatbot.add_message_to_history(request.message, full_response)
            
            # Save to MongoDB
            await message_history.save_message("user", request.message)
            await message_history.save_message("assistant", full_response)
            
            # Trim chat history in MongoDB if needed
            await ChatHistoryDB.trim_chat_history(user_id, MAX_CONVERSATION_HISTORY * 2)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'done': True, 'error': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ==================== ADDITIONAL ENDPOINTS ====================
@app.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(current_user: dict = Depends(get_current_user)):
    """Get conversation history for current user"""
    user_id = current_user["user_id"]
    
    messages_data = await ChatHistoryDB.get_chat_history(user_id)
    
    messages = [
        ChatMessage(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg.get("timestamp")
        )
        for msg in messages_data
    ]
    
    return ChatHistoryResponse(success=True, messages=messages)


@app.delete("/chat/history", response_model=GenericResponse)
async def clear_chat_history(current_user: dict = Depends(get_current_user)):
    """Clear conversation history for current user"""
    user_id = current_user["user_id"]
    
    await ChatHistoryDB.clear_chat_history(user_id)
    
    if user_id in user_message_histories:
        user_message_histories[user_id].clear()
    
    return GenericResponse(success=True, message="Chat history cleared successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)"""
    doc_count = shared_db_manager.get_collection_size() if shared_db_manager else 0
    
    return {
        "status": "healthy",
        "database_loaded": doc_count > 0,
        "total_documents": doc_count
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PC Parts Chatbot API",
        "version": "3.0.0",
        "docs": "/docs"
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)