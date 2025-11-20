from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from logic import GenieEngine
import os

app = FastAPI(title="GenieSuite API")

# Allow Frontend to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GenieEngine()

class RequestModel(BaseModel):
    goal: str
    industry: str

@app.get("/")
def read_root():
    return {"status": "System Operational", "mode": "Secure Pipeline"}

@app.post("/api/process")
async def process_request(req: RequestModel):
    try:
        result = engine.run_pipeline(req.goal, req.industry)
        
        # Generate the document immediately
        filename = f"report_{os.urandom(4).hex()}.docx"
        engine.generate_doc(result, filename)
        
        return {
            "status": "success",
            "data": result,
            "download_link": f"/api/download/{filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    