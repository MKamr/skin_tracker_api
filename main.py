from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import random
import io
from PIL import Image
import numpy as np
import cv2
from typing import Optional

app = FastAPI(title="AI-Powered Skin Tracker", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    image_data: Optional[str] = None  # Base64 encoded image
    description: Optional[str] = None

class AnalysisResponse(BaseModel):
    patch_size_percentage: float
    pigmentation_score: float
    analysis_confidence: float
    message: str

@app.get("/")
async def root():
    return {"message": "AI-Powered Skin Tracker API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "skin-tracker-api"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_skin_image(file: UploadFile = File(...)):
    """
    Analyze uploaded skin image and return patch size percentage and pigmentation score.
    This is a stub implementation that returns random values for demonstration.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Convert to PIL Image for basic processing
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # STUB AI IMPLEMENTATION
        # Generate random but realistic values for demonstration
        patch_size_percentage = round(random.uniform(5.0, 45.0), 2)
        pigmentation_score = round(random.uniform(0.1, 0.9), 3)
        analysis_confidence = round(random.uniform(0.75, 0.95), 3)
        
        # Create response message
        message = f"Analysis complete for {width}x{height} image. Patch covers {patch_size_percentage}% of analyzed area."
        
        return AnalysisResponse(
            patch_size_percentage=patch_size_percentage,
            pigmentation_score=pigmentation_score,
            analysis_confidence=analysis_confidence,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-base64", response_model=AnalysisResponse)
async def analyze_skin_base64(request: AnalysisRequest):
    """
    Alternative endpoint for base64 encoded images (useful for web frontends).
    """
    try:
        if not request.image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64 image (basic validation)
        import base64
        try:
            # Remove data URL prefix if present
            if request.image_data.startswith('data:image'):
                request.image_data = request.image_data.split(',')[1]
            
            image_data = base64.b64decode(request.image_data)
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # STUB AI IMPLEMENTATION (same as file upload)
        patch_size_percentage = round(random.uniform(5.0, 45.0), 2)
        pigmentation_score = round(random.uniform(0.1, 0.9), 3)
        analysis_confidence = round(random.uniform(0.75, 0.95), 3)
        
        message = f"Analysis complete for {width}x{height} image. Patch covers {patch_size_percentage}% of analyzed area."
        
        return AnalysisResponse(
            patch_size_percentage=patch_size_percentage,
            pigmentation_score=pigmentation_score,
            analysis_confidence=analysis_confidence,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
