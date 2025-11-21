from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from logic import GenieEngine, Agent
import os
import logging
from datetime import datetime
from typing import Optional, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenieSuite API",
    description="Advanced AI-powered business strategy assistant",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = GenieEngine()
agent = Agent(engine)

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Request Models
class RequestModel(BaseModel):
    goal: str = Field(..., min_length=5, max_length=500, description="Business goal description")
    industry: str = Field(default="General Business", max_length=100, description="Industry sector")

class BatchRequestModel(BaseModel):
    requests: List[RequestModel] = Field(..., max_items=10, description="Batch of requests")

class FeedbackModel(BaseModel):
    request_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session identifier for memory storage")
    message: str = Field(..., min_length=1, description="User message to the agent")


class MemoryActionModel(BaseModel):
    session_id: str
    action: str = Field(..., description="Action to perform: get, clear")


class ActionRequest(BaseModel):
    action: str = Field(..., description="Named action to execute, e.g., summarize, plan, search, generate_doc")
    payload: Optional[dict] = None

# Health Check
@app.get("/")
def read_root():
    return {
        "status": "System Operational",
        "mode": "Secure Pipeline",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
def health_check():
    """Detailed health check endpoint"""
    try:
        # Check if engine is initialized
        engine_status = "ready" if engine else "not_ready"
        
        # Check outputs directory
        outputs_accessible = os.path.exists("outputs") and os.access("outputs", os.W_OK)
        
        return {
            "status": "healthy",
            "engine": engine_status,
            "outputs_directory": "accessible" if outputs_accessible else "not_accessible",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Main Processing Endpoint
@app.post("/api/process")
async def process_request(req: RequestModel, background_tasks: BackgroundTasks):
    """
    Process a business goal and generate a strategy report.
    """
    try:
        logger.info(f"Processing request: goal='{req.goal[:50]}...', industry='{req.industry}'")
        
        # Validate input
        if not req.goal or len(req.goal.strip()) < 5:
            raise HTTPException(
                status_code=400,
                detail="Goal must be at least 5 characters long"
            )
        
        # Run the pipeline
        result = engine.run_pipeline(req.goal, req.industry)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate strategy. Please try again."
            )
        
        # Generate document in background
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.docx"
        
        try:
            file_path = engine.generate_doc(result, filename)
            
            # Clean up old files in background (older than 24 hours)
            background_tasks.add_task(cleanup_old_files)
            
            return {
                "status": "success",
                "data": result,
                "download_link": f"/api/download/{filename}",
                "request_id": filename.replace('.docx', ''),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Document generation failed: {str(e)}")
            # Return strategy even if document generation fails
            return {
                "status": "success",
                "data": result,
                "download_link": None,
                "warning": "Document generation failed, but strategy was generated successfully",
                "request_id": f"req_{os.urandom(8).hex()}",
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/chat")
async def chat_endpoint(chat: ChatRequest):
    """Conversational endpoint supporting session memory and intent routing."""
    session_id = chat.session_id or f"sess_{os.urandom(4).hex()}"
    try:
        result = agent.handle_message(session_id, chat.message)
        return {"status": "success", "session_id": session_id, "result": result}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memory")
async def memory_action(mem: MemoryActionModel):
    """Get or clear session memory."""
    try:
        if mem.action == "get":
            session = agent.memory.get_session(mem.session_id)
            return {"status": "success", "session": session}
        if mem.action == "clear":
            agent.memory.clear_session(mem.session_id)
            return {"status": "success", "message": "session cleared"}
        raise HTTPException(status_code=400, detail="Unknown memory action")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/action")
async def invoke_action(req: ActionRequest):
    """Invoke a specific agent action programmatically."""
    try:
        action = req.action
        payload = req.payload or {}

        # route to agent methods
        if action == "summarize":
            text = payload.get("text", "")
            return {"status": "success", "response": agent.summarize_text(text)}
        if action == "plan":
            goal = payload.get("goal", "")
            industry = payload.get("industry", "General Business")
            data = agent.engine.run_pipeline(goal, industry)
            return {"status": "success", "data": data}
        if action == "todo":
            text = payload.get("text", "")
            return {"status": "success", "response": agent.extract_action_items(text)}
        if action == "search":
            query = payload.get("query", "")
            results = agent.engine.search_web(query)
            return {"status": "success", "results": results}
        if action == "generate_doc":
            goal = payload.get("goal", "")
            industry = payload.get("industry", "General Business")
            result = agent.engine.run_pipeline(goal, industry)
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.docx"
            path = agent.engine.generate_doc(result, filename)
            return {"status": "success", "path": path, "data": result}

        raise HTTPException(status_code=400, detail="Unknown action")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Action invocation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Processing Endpoint
@app.post("/api/process/batch")
async def process_batch_request(batch: BatchRequestModel):
    """
    Process multiple requests in batch.
    """
    try:
        results = []
        for i, req in enumerate(batch.requests):
            try:
                result = engine.run_pipeline(req.goal, req.industry)
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}_{os.urandom(4).hex()}.docx"
                engine.generate_doc(result, filename)
                
                results.append({
                    "status": "success",
                    "data": result,
                    "download_link": f"/api/download/{filename}",
                    "request_id": filename.replace('.docx', '')
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "error": str(e),
                    "goal": req.goal[:50]
                })
        
        return {
            "status": "completed",
            "total": len(batch.requests),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Download Endpoint
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """
    Download a generated report file.
    """
    # Security: Prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join("outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        file_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

# List Reports Endpoint
@app.get("/api/reports")
async def list_reports(limit: int = 10):
    """
    List available reports.
    """
    try:
        if not os.path.exists("outputs"):
            return {"reports": [], "total": 0}
        
        files = []
        for filename in os.listdir("outputs"):
            if filename.endswith('.docx'):
                file_path = os.path.join("outputs", filename)
                file_stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": file_stat.st_size,
                    "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "download_link": f"/api/download/{filename}"
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "reports": files[:limit],
            "total": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback Endpoint
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """
    Submit feedback about a generated strategy.
    """
    try:
        # In a real application, you would save this to a database
        feedback_data = {
            "request_id": feedback.request_id,
            "rating": feedback.rating,
            "feedback": feedback.feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file (in production, use a database)
        feedback_file = "feedback.json"
        feedbacks = []
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedbacks = json.load(f)
        
        feedbacks.append(feedback_data)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedbacks, f, indent=2)
        
        logger.info(f"Feedback received: rating={feedback.rating}, request_id={feedback.request_id}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!"
        }
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoint
@app.get("/api/analytics")
async def get_analytics():
    """
    Get analytics data about API usage.
    """
    try:
        # Count reports
        report_count = 0
        total_size = 0
        if os.path.exists("outputs"):
            for filename in os.listdir("outputs"):
                if filename.endswith('.docx'):
                    report_count += 1
                    file_path = os.path.join("outputs", filename)
                    total_size += os.path.getsize(file_path)
        
        # Count feedback
        feedback_count = 0
        avg_rating = 0
        if os.path.exists("feedback.json"):
            with open("feedback.json", 'r') as f:
                feedbacks = json.load(f)
                feedback_count = len(feedbacks)
                if feedbacks:
                    avg_rating = sum(f.get("rating", 0) for f in feedbacks) / len(feedbacks)
        
        return {
            "total_reports": report_count,
            "total_size_bytes": total_size,
            "total_feedback": feedback_count,
            "average_rating": round(avg_rating, 2) if avg_rating > 0 else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility Functions
def cleanup_old_files():
    """
    Clean up files older than 24 hours.
    """
    try:
        if not os.path.exists("outputs"):
            return
        
        import time
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours in seconds
        
        for filename in os.listdir("outputs"):
            file_path = os.path.join("outputs", filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > max_age:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
