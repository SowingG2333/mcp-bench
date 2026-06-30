#!/usr/bin/env python3
"""
FastAPI service for LLM-Agent interaction via MCP-Bench.

Usage:
    python api_chat.py
    python api_chat.py --host 0.0.0.0 --port 8000

Endpoints:
    POST   /sessions              Create a new chat session
    POST   /sessions/{id}/chat    Send a message and get agent response
    GET    /sessions/{id}         Get session history and metadata
    DELETE /sessions/{id}         Close a session and release MCP connections
    GET    /models                List available registered models
    GET    /servers               List available MCP servers
"""
import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

from agent.chat_session import ChatSession, ChatSessionState
from llm.factory import LLMFactory
from utils.local_server_config import LocalServerConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory session store
_sessions: Dict[str, ChatSession] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    model: Optional[str] = Field(
        None, description="Registered model name. Required unless using direct endpoint."
    )
    base_url: Optional[str] = Field(None, description="Direct OpenAI-compatible base URL")
    api_key: Optional[str] = Field("", description="API key for direct endpoint")
    model_name: Optional[str] = Field(None, description="Actual model name for direct endpoint")
    provider: str = Field("openai_compatible", description="Provider type for direct endpoint")
    servers: List[str] = Field(default_factory=list, description="List of MCP server names (empty = all)")
    save_dir: str = Field("sessions", description="Directory to persist session history")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    timeout: int = Field(300, ge=30, le=600, description="Timeout in seconds")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    total_rounds: int = 0
    tool_calls: int = 0
    total_tokens: int = 0


class SessionInfoResponse(BaseModel):
    session_id: str
    model_name: str
    servers: List[str]
    created_at: float
    updated_at: float
    history: List[Dict]


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API server starting")
    yield
    logger.info("API server shutting down, closing %d session(s)", len(_sessions))
    for session in list(_sessions.values()):
        try:
            await session.close()
        except Exception as e:
            logger.warning("Error closing session %s: %s", session.session_id, e)
    _sessions.clear()


app = FastAPI(
    title="MCP-Bench Agent API",
    description="REST API for interacting with the MCP-Bench agent through LLM models.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_session(session_id: str) -> ChatSession:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "MCP-Bench Agent API",
        "docs": "/docs",
        "endpoints": {
            "models": "/models",
            "servers": "/servers",
            "create_session": "POST /sessions",
            "chat": "POST /sessions/{id}/chat",
            "get_session": "GET /sessions/{id}",
            "close_session": "DELETE /sessions/{id}",
        },
    }


@app.get("/models")
async def list_models():
    configs = LLMFactory.get_model_configs()
    return {
        "models": [
            {
                "name": name,
                "provider_type": cfg.provider_type,
                "model_name": cfg.config.get("model_name", cfg.config.get("deployment_name")),
            }
            for name, cfg in configs.items()
        ]
    }


@app.get("/servers")
async def list_servers():
    loader = LocalServerConfigLoader()
    return {"servers": sorted(loader.local_commands.keys())}


@app.post("/sessions", response_model=SessionInfoResponse, status_code=201)
async def create_session(req: CreateSessionRequest):
    try:
        if req.base_url:
            if not req.model_name:
                raise HTTPException(status_code=422, detail="model_name is required when base_url is provided")
            session = ChatSession.from_direct_endpoint(
                base_url=req.base_url,
                api_key=req.api_key or "",
                model_name=req.model_name,
                display_name=req.model or req.model_name,
                provider=req.provider,
                server_names=req.servers or None,
                save_dir=req.save_dir,
            )
        else:
            if not req.model:
                configs = LLMFactory.get_model_configs()
                if not configs:
                    raise HTTPException(status_code=400, detail="No registered models available")
                req.model = list(configs.keys())[0]
            session = ChatSession.from_env(
                model_name=req.model,
                server_names=req.servers or None,
                save_dir=req.save_dir,
            )

        await session.initialize()
        _sessions[session.session_id] = session

        return SessionInfoResponse(
            session_id=session.session_id,
            model_name=session.model_name,
            servers=session.server_names,
            created_at=session.created_at,
            updated_at=session.updated_at,
            history=session.history,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    session = _require_session(session_id)
    try:
        result = await session.chat(req.message, timeout_seconds=req.timeout)
        if "error" in result:
            # Still return a 200 with the error in the response field
            logger.warning("Chat turn produced error: %s", result.get("error"))
        return ChatResponse(
            session_id=session_id,
            response=result.get("response", ""),
            total_rounds=result.get("total_rounds", 0),
            tool_calls=result.get("tool_calls", 0),
            total_tokens=result.get("total_tokens", 0),
        )
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    session = _require_session(session_id)
    return SessionInfoResponse(
        session_id=session.session_id,
        model_name=session.model_name,
        servers=session.server_names,
        created_at=session.created_at,
        updated_at=session.updated_at,
        history=session.history,
    )


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    await session.close()
    return {"status": "closed", "session_id": session_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FastAPI server for MCP-Bench agent")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    uvicorn_run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("FastAPI/Uvicorn not installed. Please run: pip install fastapi uvicorn")
        raise SystemExit(1)
    main()
