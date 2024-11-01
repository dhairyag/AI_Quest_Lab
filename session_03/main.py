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
import matplotlib.pyplot as plt
import base64

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
        if data_type == "3d_geometry":
            if not file.filename.endswith('.off'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file format. Only .OFF files are supported for 3D geometry."
                )
        
        if data_type == "audio":
            valid_audio_types = ["audio/wav", "audio/mpeg", "audio/ogg"]
            if file.content_type not in valid_audio_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid audio format. Supported formats: WAV, MP3, OGG. Got: {file.content_type}"
                )
        
        file_content = await file.read()
        
        if data_type == "3d_geometry":
            try:
                content = file_content.decode('utf-8').strip().split('\n')
                if content[0].strip() != 'OFF':
                    raise HTTPException(status_code=400, detail="Invalid OFF file format")
                
                n_vertices, n_faces, _ = map(int, content[1].strip().split())
                vertices = []
                for i in range(n_vertices):
                    x, y, z = map(float, content[i + 2].strip().split())
                    vertices.append([x, y, z])
                vertices = np.array(vertices)
                
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.scatter(vertices[:, 0], vertices[:, 1], c='blue', alpha=0.6)
                ax.set_aspect('equal')
                ax.set_title('2D Projection (X-Y plane)')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode()
                
                data = {
                    'vertices': vertices.tolist(),
                    'projection': f'data:image/png;base64,{img_str}'
                }
                
            except Exception as e:
                logger.error(f"Error processing 3D geometry file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing 3D geometry file: {str(e)}")
        elif data_type == "audio":
            try:
                temp_file = io.BytesIO(file_content)
                try:
                    samples, sample_rate = sf.read(temp_file)
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
            
        try:
            if data_type == "3d_geometry":
                sample = {
                    'image': data['projection'],
                    'vertices_count': len(data['vertices'])
                }
            else:
                sample = show_sample(data, data_type)
            
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
        
        # Check if the final processed_data is a dict (e.g., from MFCC)
        if data_type == "audio" and isinstance(processed_data, dict):
            sample = processed_data  # Directly use the dict for spectrogram
        else:
            sample = show_sample(processed_data, data_type)
        
        return {
            "message": "Data preprocessed successfully!",
            "sample": {
                "processed": sample
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
        logger.info(f"Augmentation techniques: {request.augmentation_techniques}")
        
        # Special handling for audio data
        if data_type == "audio":
            try:
                # Extract audio data from the frontend format
                audio_data = request.data
                if isinstance(audio_data, dict):
                    if 'audio_data' in audio_data:
                        # Data is already in the correct format
                        augmented_data = audio_data
                    else:
                        raise HTTPException(status_code=400, detail="Invalid audio data format")
                else:
                    raise HTTPException(status_code=400, detail="Invalid audio data format")
            except Exception as e:
                logger.error(f"Error parsing audio data: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid audio data format")
        else:
            augmented_data = request.data

        # Apply augmentation techniques
        for technique in request.augmentation_techniques:
            try:
                augmented_data = apply_augmentation(augmented_data, data_type, technique)
            except ValueError as e:
                logger.error(f"Error in augmentation technique '{technique}': {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error in augmentation technique '{technique}': {str(e)}")
        
        # For audio, the augmented_data will already be in the correct format for show_sample
        sample = augmented_data if data_type == "audio" else show_sample(augmented_data, data_type)
        
        return {
            "message": "Data augmented successfully!",
            "sample": {
                "processed": sample
            },
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
