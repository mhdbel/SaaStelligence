"""
FastAPI application for the SAAStelligence Engine.

This module provides the HTTP API layer for the SAAStelligence pipeline,
including ad generation, performance reporting, and health monitoring.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Rate limiting
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    Limiter = None

# Internal imports
from agents.saastelligence_agent import (
    SAAStelligenceAgent,
    AgentError,
    ValidationError as AgentValidationError,
    ModelNotFoundError,
)
from config.config import CONFIG

# ============== LOGGING CONFIGURATION ==============
logging.basicConfig(
    level=getattr(logging, CONFIG.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============== RATE LIMITING SETUP ==============
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None
    logger.warning("Rate limiting not available - install slowapi")


# ============== PYDANTIC MODELS ==============
class AdGenerationRequest(BaseModel):
    """Request model for ad generation endpoint."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user query for ad generation",
        examples=["I need help automating my sales workflow"],
    )
    user_id: Optional[str] = Field(
        None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Optional user identifier for personalization and retargeting",
    )
    last_action: Optional[str] = Field(
        None,
        max_length=200,
        description="Previous user action for contextual retargeting",
        examples=["email_submitted", "form_abandoned"],
    )
    cpa_budget: Optional[float] = Field(
        None,
        gt=0,
        le=10000,
        description="Optional CPA budget override",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize and validate query input."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class AdGenerationResponse(BaseModel):
    """Response model for ad generation endpoint."""
    
    success: bool = Field(..., description="Whether the request was successful")
    data: Dict[str, Any] = Field(..., description="Pipeline results")
    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class PerformanceReportResponse(BaseModel):
    """Response model for performance report endpoint."""
    
    success: bool
    report: Dict[str, Any]
    generated_at: str


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    agent_ready: bool = Field(..., description="Whether the agent is initialized")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    success: bool = False
    error: str
    error_code: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============== LIFESPAN MANAGEMENT ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.
    
    - Initializes the SAAStelligence agent on startup
    - Performs cleanup on shutdown
    """
    # Startup
    logger.info("🚀 Starting SAAStelligence Engine...")
    
    try:
        app.state.agent = SAAStelligenceAgent()
        app.state.startup_time = time.time()
        logger.info("✅ SAAStelligence Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SAAStelligence Engine...")
    
    if hasattr(app.state, 'agent') and app.state.agent:
        # Cleanup agent resources
        if hasattr(app.state.agent, '_executor'):
            app.state.agent._executor.shutdown(wait=True)
    
    logger.info("👋 Shutdown complete")


# ============== APP INITIALIZATION ==============
app = FastAPI(
    title="SAAStelligence Engine",
    description="""
    AI-powered SaaS intelligence and ad generation API.
    
    ## Features
    - Intent detection from user queries
    - Dynamic ad copy generation
    - Smart bid adjustment
    - Conversion funnel routing
    - Performance analytics
    
    ## Authentication
    API key required for production endpoints (X-API-Key header).
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ============== MIDDLEWARE ==============
# CORS Configuration
allowed_origins = getattr(CONFIG, 'ALLOWED_ORIGINS', ["*"])
if isinstance(allowed_origins, str):
    allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"],
)

# Rate limiting middleware
if RATE_LIMITING_AVAILABLE and limiter:
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=ErrorResponse(
                error="Rate limit exceeded. Please slow down.",
                error_code="RATE_LIMIT_EXCEEDED",
                request_id=getattr(request.state, 'request_id', None),
            ).model_dump(),
        )


# ============== REQUEST TRACKING MIDDLEWARE ==============
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """
    Add request tracking, logging, and timing to all requests.
    
    - Generates unique request ID
    - Logs request/response info
    - Adds timing headers
    """
    # Generate request ID
    request_id = str(uuid.uuid4())[:12]
    request.state.request_id = request_id
    
    # Log incoming request
    start_time = time.time()
    logger.info(
        f"[{request_id}] ← {request.method} {request.url.path}",
        extra={"client_ip": request.client.host if request.client else "unknown"},
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time_ms:.2f}ms"
        
        # Log response
        logger.info(
            f"[{request_id}] → {response.status_code} ({processing_time_ms:.2f}ms)"
        )
        
        return response
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[{request_id}] ✗ Error after {processing_time_ms:.2f}ms: {e}",
            exc_info=True,
        )
        raise


# ============== GLOBAL EXCEPTION HANDLERS ==============
@app.exception_handler(AgentValidationError)
async def agent_validation_error_handler(request: Request, exc: AgentValidationError):
    """Handle agent validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_code="VALIDATION_ERROR",
            request_id=getattr(request.state, 'request_id', None),
            details=getattr(exc, 'details', None),
        ).model_dump(),
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    """Handle model not found errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Intent model not available. Please contact support.",
            error_code="MODEL_NOT_FOUND",
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    """Handle general agent errors."""
    logger.error(f"Agent error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An error occurred processing your request.",
            error_code="AGENT_ERROR",
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler.
    
    Prevents internal error details from leaking to clients.
    """
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": getattr(request.state, 'request_id', None)},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An internal error occurred. Please try again later.",
            error_code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


# ============== DEPENDENCY INJECTION ==============
def get_agent(request: Request) -> SAAStelligenceAgent:
    """
    Dependency to retrieve the agent instance.
    
    Raises HTTPException if agent is not available.
    """
    agent = getattr(request.app.state, 'agent', None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized",
        )
    return agent


def get_request_id(request: Request) -> str:
    """Dependency to retrieve the current request ID."""
    return getattr(request.state, 'request_id', 'unknown')


# ============== API KEY AUTHENTICATION (Optional) ==============
async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key for protected endpoints.
    
    Set REQUIRE_API_KEY=true and API_KEY in config to enable.
    """
    if not getattr(CONFIG, 'REQUIRE_API_KEY', False):
        return None
    
    api_key = request.headers.get("X-API-Key")
    expected_key = getattr(CONFIG, 'API_KEY', None)
    
    if not expected_key:
        logger.warning("API key authentication enabled but no API_KEY configured")
        return None
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != expected_key:
        logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return api_key


# ============== ENDPOINTS ==============

# ----- System Endpoints -----

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Check the health status of the API and its components.",
)
async def health_check(request: Request):
    """
    Health check endpoint for container orchestration and monitoring.
    
    Returns the current status of the API and its components.
    """
    agent = getattr(request.app.state, 'agent', None)
    startup_time = getattr(request.app.state, 'startup_time', None)
    
    components = {
        "api": "healthy",
        "agent": "healthy" if agent else "unavailable",
    }
    
    # Check if model is loaded
    if agent:
        try:
            if agent._cached_model is not None:
                components["intent_model"] = "loaded"
            else:
                components["intent_model"] = "not_loaded"
        except Exception:
            components["intent_model"] = "unknown"
        
        # Check ad generation capability
        components["ad_generation"] = "enabled" if agent._ad_chain else "disabled"
    
    overall_status = "healthy" if all(
        v in ("healthy", "loaded", "enabled", "not_loaded") 
        for v in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        agent_ready=agent is not None,
        components=components,
    )


@app.get(
    "/",
    tags=["System"],
    summary="Root endpoint",
    include_in_schema=False,
)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SAAStelligence Engine",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health",
    }


# ----- Core API Endpoints -----

@app.post(
    "/api/v1/generate-ad",
    response_model=AdGenerationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["Ad Generation"],
    summary="Generate an ad based on user query",
    description="""
    Execute the complete SAAStelligence pipeline:
    
    1. **Intent Detection**: Classify user query into intent categories
    2. **Ad Generation**: Create personalized ad copy (requires OpenAI API key)
    3. **Funnel Routing**: Determine optimal conversion funnel
    4. **Bid Adjustment**: Calculate optimal bid based on predicted CVR
    5. **Retargeting**: Generate retargeting URL if applicable
    """,
)
async def generate_ad(
    request: Request,
    payload: AdGenerationRequest,
    agent: SAAStelligenceAgent = Depends(get_agent),
    request_id: str = Depends(get_request_id),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Generate an AI-powered advertisement based on the provided query.
    
    The pipeline analyzes user intent, generates relevant ad copy,
    determines the optimal funnel, and calculates bid adjustments.
    """
    start_time = time.time()
    
    logger.info(
        f"[{request_id}] Processing ad generation",
        extra={
            "query_length": len(payload.query),
            "has_user_id": payload.user_id is not None,
            "has_last_action": payload.last_action is not None,
        },
    )
    
    try:
        # Use async method to avoid blocking the event loop
        result = await agent.run_async(
            user_query=payload.query,
            user_id=payload.user_id,
            last_action=payload.last_action,
            cpa_budget=payload.cpa_budget,
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"[{request_id}] Ad generation complete",
            extra={
                "intent": result.get("intent"),
                "confidence": result.get("intent_confidence"),
                "processing_time_ms": processing_time_ms,
            },
        )
        
        return AdGenerationResponse(
            success=True,
            data=dict(result),
            request_id=request_id,
            processing_time_ms=round(processing_time_ms, 2),
        )
        
    except AgentValidationError:
        # Re-raise to be handled by specific handler
        raise
    except ModelNotFoundError:
        # Re-raise to be handled by specific handler
        raise
    except Exception as e:
        logger.error(
            f"[{request_id}] Error generating ad: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate ad. Please try again.",
        )


@app.get(
    "/api/v1/report",
    response_model=PerformanceReportResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["Analytics"],
    summary="Get performance report",
    description="Retrieve aggregated performance metrics including CTR, CVR, CPA, and lead counts.",
)
async def get_report(
    request: Request,
    agent: SAAStelligenceAgent = Depends(get_agent),
    request_id: str = Depends(get_request_id),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Retrieve the performance metrics and analytics report.
    
    Metrics include:
    - **CTR**: Click-through rate
    - **CVR**: Conversion rate
    - **CPA**: Cost per acquisition
    - **Leads**: Total lead count
    """
    try:
        # Use async method
        report = await agent.report_performance_async()
        
        from datetime import datetime
        
        return PerformanceReportResponse(
            success=True,
            report=report,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error generating report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report",
        )


# ----- Training Endpoint (Protected) -----

class TrainingRequest(BaseModel):
    """Request model for training endpoint."""
    
    conversion_data: list[Dict[str, str]] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of conversion records for training",
    )


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    
    success: bool
    records_trained: int
    message: str


@app.post(
    "/api/v1/train",
    response_model=TrainingResponse,
    tags=["Training"],
    summary="Submit training data",
    description="Submit new conversion data to retrain the intent model.",
)
async def train_model(
    request: Request,
    payload: TrainingRequest,
    agent: SAAStelligenceAgent = Depends(get_agent),
    request_id: str = Depends(get_request_id),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Retrain the intent model with new conversion feedback data.
    
    Each record must contain: query_text, intent, converted
    Optional fields: clicks, impressions, cost
    """
    try:
        result = agent.train_from_feedback(payload.conversion_data)
        
        return TrainingResponse(
            success=True,
            records_trained=result["records_trained"],
            message="Model training initiated successfully",
        )
        
    except AgentValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[{request_id}] Training error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process training data",
        )


# ============== RATE LIMITING DECORATORS ==============
# Apply rate limiting if available
if RATE_LIMITING_AVAILABLE and limiter:
    generate_ad = limiter.limit("30/minute")(generate_ad)
    get_report = limiter.limit("10/minute")(get_report)
    train_model = limiter.limit("5/minute")(train_model)


# ============== MAIN ==============
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "web.app:app",
        host=getattr(CONFIG, 'HOST', "0.0.0.0"),
        port=getattr(CONFIG, 'PORT', 8000),
        reload=getattr(CONFIG, 'DEBUG', False),
        log_level=getattr(CONFIG, 'LOG_LEVEL', "info").lower(),
        access_log=True,
    )
