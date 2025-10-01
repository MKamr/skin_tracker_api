from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import random
import io
import base64
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Tuple
import math

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
    detected_patches: int
    average_patch_size: float
    skin_tone_variance: float

@app.get("/")
async def root():
    return {"message": "AI-Powered Skin Tracker API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "skin-tracker-api"}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for skin analysis.
    """
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize if too large (for performance)
    height, width = cv_image.shape[:2]
    if width > 1000 or height > 1000:
        scale = min(1000/width, 1000/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        cv_image = cv2.resize(cv_image, (new_width, new_height))
    
    return cv_image

def detect_skin_patches(cv_image: np.ndarray) -> Tuple[float, float, int, float, float]:
    """
    Detect skin patches using OpenCV and return analysis results.
    
    Returns:
        patch_size_percentage: Percentage of image covered by patches
        pigmentation_score: Average pigmentation intensity (0-1)
        detected_patches: Number of distinct patches found
        average_patch_size: Average size of patches in pixels
        skin_tone_variance: Variance in skin tone across the image
    """
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    
    # Create skin mask using HSV color range
    # These ranges work well for most skin tones
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of skin regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours (noise)
    min_area = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Calculate total skin area
    total_skin_area = cv2.countNonZero(skin_mask)
    total_image_area = cv_image.shape[0] * cv_image.shape[1]
    
    if total_skin_area == 0:
        return 0.0, 0.0, 0, 0.0, 0.0
    
    # Analyze pigmentation using LAB color space
    # L channel represents lightness, which is good for pigmentation analysis
    l_channel = lab[:, :, 0]
    
    # Calculate pigmentation variance within skin regions
    skin_pixels = l_channel[skin_mask > 0]
    if len(skin_pixels) > 0:
        pigmentation_variance = np.var(skin_pixels) / 255.0  # Normalize to 0-1
        average_lightness = np.mean(skin_pixels) / 255.0
        pigmentation_score = 1.0 - average_lightness  # Darker = higher pigmentation
    else:
        pigmentation_variance = 0.0
        pigmentation_score = 0.0
    
    # Calculate patch statistics
    detected_patches = len(valid_contours)
    
    if detected_patches > 0:
        patch_areas = [cv2.contourArea(c) for c in valid_contours]
        average_patch_size = np.mean(patch_areas)
        
        # Calculate total patch area
        total_patch_area = sum(patch_areas)
        patch_size_percentage = (total_patch_area / total_image_area) * 100
    else:
        average_patch_size = 0.0
        patch_size_percentage = 0.0
    
    # Calculate skin tone variance across the entire image
    if len(skin_pixels) > 0:
        skin_tone_variance = np.std(skin_pixels) / 255.0
    else:
        skin_tone_variance = 0.0
    
    return (
        round(patch_size_percentage, 2),
        round(pigmentation_score, 3),
        detected_patches,
        round(average_patch_size, 1),
        round(skin_tone_variance, 3)
    )

def calculate_analysis_confidence(cv_image: np.ndarray, detected_patches: int, skin_tone_variance: float) -> float:
    """
    Calculate confidence score based on image quality and analysis results.
    """
    height, width = cv_image.shape[:2]
    
    # Base confidence on image size (larger images = higher confidence)
    size_confidence = min(1.0, (width * height) / (500 * 500))
    
    # Confidence based on number of detected patches
    patch_confidence = min(1.0, detected_patches / 10.0) if detected_patches > 0 else 0.5
    
    # Confidence based on skin tone variance (some variance is good)
    variance_confidence = min(1.0, skin_tone_variance * 2) if skin_tone_variance > 0 else 0.3
    
    # Combine confidence factors
    overall_confidence = (size_confidence * 0.4 + patch_confidence * 0.3 + variance_confidence * 0.3)
    
    return round(max(0.5, min(0.95, overall_confidence)), 3)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_skin_image(file: UploadFile = File(...)):
    """
    Analyze uploaded skin image using OpenCV and return real patch analysis results.
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
        
        # Preprocess image for analysis
        cv_image = preprocess_image(image)
        
        # Perform real skin patch analysis
        patch_size_percentage, pigmentation_score, detected_patches, average_patch_size, skin_tone_variance = detect_skin_patches(cv_image)
        
        # Calculate analysis confidence
        analysis_confidence = calculate_analysis_confidence(cv_image, detected_patches, skin_tone_variance)
        
        # Create detailed response message
        message = f"Analysis complete for {width}x{height} image. Found {detected_patches} skin patches covering {patch_size_percentage}% of the area."
        
        return AnalysisResponse(
            patch_size_percentage=patch_size_percentage,
            pigmentation_score=pigmentation_score,
            analysis_confidence=analysis_confidence,
            message=message,
            detected_patches=detected_patches,
            average_patch_size=average_patch_size,
            skin_tone_variance=skin_tone_variance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-base64", response_model=AnalysisResponse)
async def analyze_skin_base64(request: AnalysisRequest):
    """
    Alternative endpoint for base64 encoded images using real OpenCV analysis.
    """
    try:
        if not request.image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode base64 image
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
        
        # Preprocess image for analysis
        cv_image = preprocess_image(image)
        
        # Perform real skin patch analysis
        patch_size_percentage, pigmentation_score, detected_patches, average_patch_size, skin_tone_variance = detect_skin_patches(cv_image)
        
        # Calculate analysis confidence
        analysis_confidence = calculate_analysis_confidence(cv_image, detected_patches, skin_tone_variance)
        
        # Create detailed response message
        message = f"Analysis complete for {width}x{height} image. Found {detected_patches} skin patches covering {patch_size_percentage}% of the area."
        
        return AnalysisResponse(
            patch_size_percentage=patch_size_percentage,
            pigmentation_score=pigmentation_score,
            analysis_confidence=analysis_confidence,
            message=message,
            detected_patches=detected_patches,
            average_patch_size=average_patch_size,
            skin_tone_variance=skin_tone_variance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
