import io
import base64
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import processing functions from llamang.py
from llamang import extract_document_info, preprocess, preview

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llamang-api")

app = FastAPI(
    title="LlamaNG Document Analysis API",
    description="API for extracting structured information from document images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_bboxes(results: List[Dict[str, Any]], img_width: int, img_height: int) -> Dict[str, Any]:
    """
    Convert absolute bbox coordinates to normalized format and structure the response
    
    Args:
        results: List of extracted results with bboxes
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        Dictionary with normalized fields and image information
    """
    fields = []
    
    for i, item in enumerate(results):
        if "bbox" not in item:
            continue
            
        # Original bbox is [x1, y1, x2, y2]
        x1, y1, x2, y2 = item["bbox"]
        
        # remove border: 10
        x1 -= 10
        y1 -= 10
        x2 -= 10
        y2 -= 10

        # Calculate normalized coordinates
        x = x1 / img_width
        y = y1 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        field = {
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "id": i,
            "name": item["key"],
            "psm": 1  # Using default PSM value
        }
        
        fields.append(field)
    
    return {
        "fields": fields,
        "height": img_height,
        "width": img_width,
        "name": "Generated Template"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/")
async def process_document(
    file: UploadFile = File(...),
    autoscale: bool = Form(False),
    include_preview: bool = Form(False),
):
    """
    Process a document image and extract information
    
    - **file**: Document image file
    - **autoscale**: Whether to automatically scale the image
    - **include_preview**: Whether to include the preview image in base64 format
    """
    try:
        logger.info(f"Processing document: {file.filename}, autoscale={autoscale}")
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Store original dimensions
        original_height, original_width = original_image.shape[:2]
        image = original_image.copy()
        
        used_height = original_height
        used_width = original_width
        # Apply autoscaling if requested
        if autoscale:
            used_height = h = image.shape[0]
            used_width = w = image.shape[1]
            dpi = 200
            scale = 1

            if w < h:
                scale = dpi / (w / 8.5)
            else:
                scale = dpi / (h / 11)

            if scale != 1:
                logger.info(f"Autoscaling image with scale factor {scale}")
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
        # Preprocess the image
        logger.info("Preprocessing image")
        image = preprocess._preprocess(preprocess.steps, image)
        
        # Extract document information
        logger.info("Extracting document information")
        result = await extract_document_info(image)
        
        # Normalize bounding boxes and structure the response
        normalized_response = normalize_bboxes(result, used_width, used_height)
        
        # Add metadata
        response = {
            **normalized_response,
            "metadata": {
                "filename": file.filename,
                "autoscaled": autoscale
            }
        }
        
        # Include preview if requested
        if include_preview:
            logger.info("Generating preview image")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                preview_path = tmp_file.name
            
            # Generate preview image
            preview(image, result, preview_path)
            
            # Read and encode the preview image
            with open(preview_path, "rb") as img_file:
                preview_bytes = img_file.read()
            
            # Include base64 encoded image in response
            response["preview"] = base64.b64encode(preview_bytes).decode('utf-8')
            
            # Clean up the temporary file
            try:
                os.unlink(preview_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")
        
        logger.info("Processing completed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True)
