from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Any, Union
from preprocessing import apply_preprocessing
from augmentation import apply_augmentation
from utils import load_data, show_sample
import json
import logging
from pydantic import BaseModel, Field
import soundfile as sf
import io
import numpy as np

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
        if data_type == "audio":
            # Validate audio file type
            valid_audio_types = ["audio/wav", "audio/mpeg", "audio/ogg"]
            if file.content_type not in valid_audio_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid audio format. Supported formats: WAV, MP3, OGG. Got: {file.content_type}"
                )
        
        file_content = await file.read()
        
        if data_type == "audio":
            try:
                # Save the uploaded file temporarily
                temp_file = io.BytesIO(file_content)
                # Read audio file using soundfile
                try:
                    samples, sample_rate = sf.read(temp_file)
                    print(f"Audio file read successfully - Sample rate: {sample_rate}, Shape: {samples.shape}")
                    
                    # Validate audio data
                    if len(samples) == 0:
                        raise HTTPException(status_code=400, detail="Audio file is empty")
                    
                    data = (samples, sample_rate)
                except sf.LibsndfileError as e:
                    raise HTTPException(
                        status_code=400,
                        detail="Error reading audio file. Make sure it's a valid WAV file."
                    )
                    
            except Exception as e:
                logger.error(f"Error processing audio file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing audio file: {str(e)}")
        else:
            data = load_data(file_content, data_type)
            
        # Process the sample
        try:
            sample = show_sample(data, data_type)
            
            # For audio, verify the sample contains required data
            if data_type == "audio":
                if not isinstance(sample, dict) or 'audio_data' not in sample:
                    raise HTTPException(status_code=500, detail="Invalid audio sample format")
                print(f"Audio sample prepared - Base64 length: {len(sample['audio_data'])}")
                
            return {
                "message": "Data uploaded successfully!",
                "sample": {
                    "original": sample,
                    "processed": None
                },
                "data_type": data_type
            }
        except Exception as e:
            logger.error(f"Error preparing sample: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error preparing sample: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class PreprocessRequest(BaseModel):
    data: Union[str, Dict[str, Any]]  # Can be either string or dictionary
    preprocessing_steps: List[str] = Field(default_factory=list)

class AugmentRequest(BaseModel):
    data: Union[str, Dict[str, Any]]  # Can be either string or dictionary
    augmentation_techniques: List[str] = Field(default_factory=list)

# Preprocessing endpoint
@app.post("/preprocess/{data_type}")
async def preprocess_data(data_type: str, request: PreprocessRequest):
    """
    Applies preprocessing steps to the data.
    """
    try:
        logger.info(f"Received preprocess request for {data_type}")
        logger.info(f"Preprocessing steps: {request.preprocessing_steps}")
        
        if data_type == "audio":
            try:
                # Extract audio data from the frontend format
                audio_data = request.data
                if isinstance(audio_data, dict):
                    if 'audio_data' in audio_data:
                        # Data is already in the correct format
                        processed_data = audio_data
                    else:
                        raise HTTPException(status_code=400, detail="Invalid audio data format")
                else:
                    raise HTTPException(status_code=400, detail="Invalid audio data format")
            except Exception as e:
                logger.error(f"Error parsing audio data: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid audio data format")
        else:
            processed_data = request.data

        # Apply preprocessing steps
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
