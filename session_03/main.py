from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any
from preprocessing import apply_preprocessing
from augmentation import apply_augmentation
from utils import load_data, show_sample
import json
import logging
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Data loading endpoint
@app.post("/upload/{data_type}")
async def upload_data(data_type: str, file: UploadFile = File(...)):
    """
    Uploads a data file.
    
    Args:
        data_type: Type of data (e.g., "text", "audio", "image").
        file: Uploaded data file.
    
    Returns:
        Dictionary with success message and sample.
    """
    try:
        file_content = await file.read()
        data = load_data(file_content, data_type)
        return {
            "message": "Data uploaded successfully!",
            "sample": {
                "original": show_sample(data, data_type),
                "processed": None  # No processing at this stage
            },
            "data_type": data_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PreprocessRequest(BaseModel):
    data: str
    preprocessing_steps: List[str] = Field(default_factory=list)

class AugmentRequest(BaseModel):
    data: str
    augmentation_techniques: List[str] = Field(default_factory=list)

# Preprocessing endpoint
@app.post("/preprocess/{data_type}")
async def preprocess_data(data_type: str, request: PreprocessRequest):
    """
    Applies preprocessing steps to the data.
    """
    try:
        logger.info(f"Received preprocess request for {data_type}")
        logger.info(f"Data: {request.data[:100]}...")  # Log first 100 characters of data
        logger.info(f"Preprocessing steps: {request.preprocessing_steps}")
        
        processed_data = request.data
        for step in request.preprocessing_steps:
            try:
                processed_data = apply_preprocessing(processed_data, data_type, step)
            except ValueError as e:
                logger.error(f"Error in preprocessing step '{step}': {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error in preprocessing step '{step}': {str(e)}")
        return {
            "message": "Data preprocessed successfully!",
            "sample": {
                "processed": show_sample(processed_data, data_type)
            },
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Augmentation endpoint
@app.post("/augment/{data_type}")
async def augment_data(data_type: str, request: AugmentRequest):
    """
    Applies data augmentation techniques.
    """
    try:
        logger.info(f"Received augment request for {data_type}")
        logger.info(f"Data: {request.data[:100]}...")  # Log first 100 characters of data
        logger.info(f"Augmentation techniques: {request.augmentation_techniques}")
        
        augmented_data = request.data
        for technique in request.augmentation_techniques:
            try:
                augmented_data = apply_augmentation(augmented_data, data_type, technique)
            except ValueError as e:
                logger.error(f"Error in augmentation technique '{technique}': {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error in augmentation technique '{technique}': {str(e)}")
        return {
            "message": "Data augmented successfully!",
            "sample": {
                "processed": show_sample(augmented_data, data_type)
            },
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
