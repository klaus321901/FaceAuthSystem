from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging

from api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Face Recognition Authentication System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    """Serve main page"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    print("=" * 60)
    print("🔐 Face Recognition Authentication System")
    print("=" * 60)
    print("Initializing...")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
