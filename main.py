"""
TechGear Electronics - Customer Support Chatbot API
FastAPI application with authentication and chatbot endpoint.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_agent import process_query, get_compiled_graph
from auth import (
    UserCreate, UserLogin, Token, User,
    create_user, authenticate_user, create_access_token,
    decode_token, get_user_by_email, init_database
)

# Load environment variables
load_dotenv()

# Security
security = HTTPBearer()


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The customer's question or message",
        json_schema_extra={"example": "What is the price of SmartWatch Pro X?"}
    )


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    response: str = Field(
        ...,
        description="The chatbot's response to the query"
    )
    category: str = Field(
        ...,
        description="The classified category of the query (PRODUCTS, RETURNS, GENERAL, ESCALATE)"
    )
    needs_escalation: bool = Field(
        default=False,
        description="Whether the query requires human escalation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "The SmartWatch Pro X is priced at ₹15,999. It features heart rate monitoring, GPS, 7-day battery life, and is water resistant up to 50m.",
                "category": "PRODUCTS",
                "needs_escalation": False
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    detail: str = Field(..., description="Error message")


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


# ============================================================================
# Authentication Dependency
# ============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Validate JWT token and return current user."""
    token = credentials.credentials
    token_data = decode_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_email(token_data.email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 Starting TechGear Electronics Chatbot API...")
    
    # Initialize database
    init_database()
    print("✅ User database initialized.")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  WARNING: GOOGLE_API_KEY not found in environment variables.")
    else:
        print("✅ Google API key found.")
    
    # Pre-compile the workflow graph
    try:
        get_compiled_graph()
        print("✅ LangGraph workflow compiled successfully.")
    except Exception as e:
        print(f"⚠️  WARNING: Could not compile workflow: {e}")
    
    print("✅ API is ready to accept requests.")
    print("📚 Swagger docs available at: http://localhost:8000/docs")
    print("🌐 Frontend available at: http://localhost:8000")
    
    yield
    
    print("👋 Shutting down TechGear Electronics Chatbot API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="TechGear Electronics Customer Support Chatbot",
    description="""
## Overview
A RAG-powered customer support chatbot for TechGear Electronics with user authentication.

## Authentication
Register an account or login to access the chat functionality.
All chat endpoints require a valid JWT token.

## Features
- **User Authentication**: Secure signup/login with JWT tokens
- **Intelligent Query Classification**: Automatically categorizes queries
- **RAG-Powered Responses**: Context-aware answers from knowledge base
- **Human Escalation**: Routes complex issues to human support
    """,
    version="2.0.0",
    contact={
        "name": "TechGear Electronics Support",
        "email": "support@techgear.com",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Static Files & Frontend
# ============================================================================

# Serve static files (frontend)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the frontend HTML page."""
    return FileResponse("static/index.html")


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"]
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="TechGear Electronics Chatbot API",
        version="2.0.0"
    )


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post(
    "/auth/signup",
    response_model=Token,
    summary="Register a new user",
    tags=["Authentication"]
)
async def signup(user_data: UserCreate):
    """
    Create a new user account.
    
    - **email**: Valid email address
    - **password**: Password (min 6 characters)
    - **name**: User's full name
    """
    # Check if user already exists
    existing_user = get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = create_user(user_data.email, user_data.name, user_data.password)
    
    # Generate token
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": user["id"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_name=user["name"],
        user_email=user["email"]
    )


@app.post(
    "/auth/login",
    response_model=Token,
    summary="Login to existing account",
    tags=["Authentication"]
)
async def login(user_data: UserLogin):
    """
    Login with email and password.
    
    - **email**: Registered email address
    - **password**: Account password
    """
    user = authenticate_user(user_data.email, user_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": user["id"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_name=user["name"],
        user_email=user["email"]
    )


@app.get(
    "/auth/me",
    response_model=User,
    summary="Get current user info",
    tags=["Authentication"]
)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get the currently authenticated user's information."""
    return User(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=str(current_user["created_at"])
    )


# ============================================================================
# Chat Endpoints (Protected)
# ============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response", "model": ChatResponse},
        401: {"description": "Unauthorized", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Chat with the support bot (requires authentication)",
    tags=["Chat"]
)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Process a customer query through the chatbot workflow.
    
    **Requires authentication** - Include Bearer token in Authorization header.
    
    - **query**: The customer's question or message (1-1000 characters)
    """
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="Google API key not configured."
            )
        
        result = process_query(request.query)
        
        return ChatResponse(
            response=result["response"],
            category=result["category"],
            needs_escalation=result["needs_escalation"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


# ============================================================================
# Run with: uvicorn main:app --reload
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
