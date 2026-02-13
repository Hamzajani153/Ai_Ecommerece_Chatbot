from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ==================== AUTH MODELS ====================
class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None


# ==================== CHAT MODELS ====================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ChatHistoryResponse(BaseModel):
    success: bool
    messages: List[ChatMessage]


# ==================== ADMIN MODELS ====================
class AdminLoadDataResponse(BaseModel):
    success: bool
    message: str
    files_processed: int
    total_chunks: int
    files_loaded: List[str]
    errors: Optional[List[str]] = None


# ==================== GENERIC RESPONSE ====================
class GenericResponse(BaseModel):
    success: bool
    message: str