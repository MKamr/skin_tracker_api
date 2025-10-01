# AI-Powered Skin Tracker - FastAPI Backend

A FastAPI-based backend service for analyzing skin images and tracking pigmentation patches.

## Features

- **Image Analysis**: Upload skin images for analysis
- **Patch Detection**: Calculate patch size percentage and pigmentation scores
- **Multiple Input Methods**: Support for file uploads and base64 encoded images
- **CORS Enabled**: Ready for frontend integration
- **Health Check**: Built-in health monitoring endpoint

## API Endpoints

### GET /
- **Description**: Root endpoint with service information
- **Response**: Service name and status

### GET /health
- **Description**: Health check endpoint
- **Response**: Service health status

### POST /analyze
- **Description**: Analyze uploaded skin image
- **Input**: Image file (multipart/form-data)
- **Response**: Analysis results with patch size percentage and pigmentation score

### POST /analyze-base64
- **Description**: Analyze base64 encoded skin image
- **Input**: JSON with base64 image data
- **Response**: Analysis results with patch size percentage and pigmentation score

## Response Format

```json
{
  "patch_size_percentage": 25.5,
  "pigmentation_score": 0.75,
  "analysis_confidence": 0.89,
  "message": "Analysis complete for 800x600 image. Patch covers 25.5% of analyzed area."
}
```

## Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server**:
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## Testing with Postman

1. **Test health endpoint**:
   - Method: GET
   - URL: http://localhost:8000/health

2. **Test image analysis**:
   - Method: POST
   - URL: http://localhost:8000/analyze
   - Body: form-data
   - Key: file (type: File)
   - Value: Select an image file

## Deployment on Render

1. **Push code to GitHub**

2. **Deploy on Render**:
   - Connect your GitHub repository
   - Use the provided Procfile
   - The service will be available at: `https://your-app-name.onrender.com`

## Current Implementation

This is a **stub implementation** that returns random but realistic values for demonstration purposes. The actual AI logic for skin analysis will be implemented in Week 3.

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Pillow**: Python Imaging Library for image processing
- **OpenCV**: Computer vision library (ready for Week 3 implementation)
- **NumPy**: Numerical computing library
- **Pydantic**: Data validation using Python type annotations

## Next Steps (Week 3)

- Implement OpenCV-based skin patch detection
- Calculate actual patch area using pixel analysis
- Improve pigmentation scoring algorithm
- Add more sophisticated image preprocessing
