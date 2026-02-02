"""
TechGear Electronics - Authentication Module
JWT-based authentication with SQLite user database.
Using bcrypt directly instead of passlib for compatibility.
"""

import os
import sqlite3
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "techgear-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
DATABASE_PATH = "users.db"


# ============================================================================
# Pydantic Models
# ============================================================================

class UserCreate(BaseModel):
    """Model for user registration."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    name: str = Field(..., min_length=2, max_length=100, description="User's full name")


class UserLogin(BaseModel):
    """Model for user login."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class Token(BaseModel):
    """Model for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_email: str


class TokenData(BaseModel):
    """Model for decoded token data."""
    email: Optional[str] = None
    user_id: Optional[int] = None


class User(BaseModel):
    """Model for user data."""
    id: int
    email: str
    name: str
    created_at: str


# ============================================================================
# Database Functions
# ============================================================================

def init_database():
    """Initialize the SQLite database with users table."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user from database by email."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, password_hash, created_at FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "email": row[1],
            "name": row[2],
            "password_hash": row[3],
            "created_at": row[4]
        }
    return None


def create_user(email: str, name: str, password: str) -> dict:
    """Create a new user in the database."""
    # Hash password using bcrypt directly
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
        (email, name, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    
    return {
        "id": user_id,
        "email": email,
        "name": name
    }


# ============================================================================
# Authentication Functions
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None:
            return None
            
        return TokenData(email=email, user_id=user_id)
    except JWTError:
        return None


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate a user by email and password."""
    user = get_user_by_email(email)
    
    if not user:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user


# Initialize database on module load
init_database()
