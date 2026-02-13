from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = "pc_parts_chatbot"

# Global MongoDB client
mongo_client: Optional[AsyncIOMotorClient] = None
database = None


async def connect_to_mongo():
    """Connect to MongoDB Atlas"""
    global mongo_client, database
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URI)
        database = mongo_client[DATABASE_NAME]
        
        # Create indexes
        await database.users.create_index("email", unique=True)
        await database.chat_history.create_index("user_id")
        
        print(" Connected to MongoDB")
    except Exception as e:
        print(f" Error connecting to MongoDB: {str(e)}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("🔌 MongoDB connection closed")


def get_database():
    """Get database instance"""
    return database


# ==================== USER OPERATIONS ====================
class UserDB:
    """User database operations"""
    
    @staticmethod
    async def create_user(email: str, hashed_password: str) -> Optional[str]:
        """
        Create a new user
        Returns user_id if successful, None if email already exists
        """
        try:
            user_doc = {
                "email": email.lower(),
                "password": hashed_password,
                "created_at": datetime.utcnow()
            }
            
            result = await database.users.insert_one(user_doc)
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            return None
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return None
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = await database.users.find_one({"email": email.lower()})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = await database.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None


# ==================== CHAT HISTORY OPERATIONS ====================
class ChatHistoryDB:
    """Chat history database operations"""
    
    @staticmethod
    async def get_chat_history(user_id: str) -> List[Dict[str, Any]]:
        """Get all chat messages for a user"""
        try:
            chat_doc = await database.chat_history.find_one({"user_id": user_id})
            
            if chat_doc and "messages" in chat_doc:
                return chat_doc["messages"]
            return []
            
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []
    
    @staticmethod
    async def add_message(user_id: str, role: str, content: str):
        """Add a message to chat history"""
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            }
            
            # Update or create chat history document
            await database.chat_history.update_one(
                {"user_id": user_id},
                {
                    "$push": {"messages": message},
                    "$set": {"updated_at": datetime.utcnow()},
                    "$setOnInsert": {"created_at": datetime.utcnow()}
                },
                upsert=True
            )
            
        except Exception as e:
            print(f"Error adding message: {str(e)}")
    
    @staticmethod
    async def clear_chat_history(user_id: str):
        """Clear all chat history for a user"""
        try:
            await database.chat_history.delete_one({"user_id": user_id})
        except Exception as e:
            print(f"Error clearing chat history: {str(e)}")
    
    @staticmethod
    async def trim_chat_history(user_id: str, max_messages: int = 20):
        """Keep only the most recent messages"""
        try:
            chat_doc = await database.chat_history.find_one({"user_id": user_id})
            
            if chat_doc and "messages" in chat_doc:
                messages = chat_doc["messages"]
                if len(messages) > max_messages:
                    # Keep only recent messages
                    trimmed_messages = messages[-max_messages:]
                    await database.chat_history.update_one(
                        {"user_id": user_id},
                        {"$set": {"messages": trimmed_messages}}
                    )
        except Exception as e:
            print(f"Error trimming chat history: {str(e)}")