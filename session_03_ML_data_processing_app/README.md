# DataAlchemy: Data Processing App

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## Overview

The **DataAlchemy** is a robust FastAPI-based web application designed to handle, preprocess, and augment various types of data, including text, audio, image, and 3D geometry. This application provides a seamless interface for uploading data, applying customizable preprocessing steps, and performing augmentation techniques to enhance data quality and diversity. It is ideal for developers, data scientists, and researchers looking to visualize their data processing effects.

## Features
- **Multi-Type Data Support:** Handle text, audio, image, and 3D geometry data seamlessly.
- **Data Upload:** Secure endpoints for uploading different data types with validation.
- **Preprocessing:** Apply a range of preprocessing steps tailored to each data type.
- **Augmentation:** Implement various augmentation techniques to expand and enhance datasets.
- **Tooltips:** ℹ️ Provide tooltips for each preprocessing and augmentation step to explain their effects.
- **Visualization:** Visualize 3D geometry data and audio spectrograms for better insights.
- **API Documentation:** Interactive API documentation powered by FastAPI.
- **Logging:** Comprehensive logging for monitoring and debugging purposes.

## Architecture

The application follows a modular architecture, promoting scalability and maintainability. The primary components include:

- **FastAPI Backend (`main.py`):** Handles API endpoints for data upload, preprocessing, and augmentation.
- **Utility Modules (`utils.py`, `preprocessing.py`, `augmentation.py`):** Encapsulate core functionalities for data handling and processing.
- **Frontend (`index.html`, `styles.css`, `script.js`):** Provides a user-friendly interface for interacting with the application.
- **Configuration (`requirements.txt`):** Lists all necessary dependencies for the project.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the Repository**
    ```bash
    git clone git@github.com:dhairyag/AI_Quest_Lab.git
    cd session_03_ML_data_processing_app
    ```

2. **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```bash
    uvicorn main:app --reload
    ```

5. **Access the App**
    Open your browser and navigate to `http://127.0.0.1:8000` to access the frontend interface.

## Usage

1. **Upload Data**
    - Navigate to the "Upload Data" section.
    - Select the data type (Text, Audio, Image, 3D Geometry).
    - Choose a file adhering to the supported formats.
    - Click the "Upload" button to submit.

2. **Process Data**
    - After uploading, select the desired preprocessing or augmentation steps.
    - Click the "Apply" button to execute the selected operations.
    - View the original and processed data in the display section.

## API Endpoints

### 1. Home
- **Endpoint:** `/`
- **Method:** `GET`
- **Description:** Serves the main HTML page.

### 2. Upload Data
- **Endpoint:** `/upload/{data_type}`
- **Method:** `POST`
- **Description:** Uploads a data file of the specified type.
- **Parameters:**
    - `data_type` (path): Type of data (`text`, `audio`, `image`, `3d_geometry`).
- **Request Body:** `multipart/form-data` with the file.
- **Responses:**
    - `200 OK`: Successful upload with a sample of the data.
    - `400 Bad Request`: Invalid file format or processing error.
    - `500 Internal Server Error`: Server-side error.

### 3. Preprocess Data
- **Endpoint:** `/preprocess/{data_type}`
- **Method:** `POST`
- **Description:** Applies preprocessing steps to the uploaded data.
- **Parameters:**
    - `data_type` (path): Type of data.
- **Request Body:** JSON with `data` and `preprocessing_steps`.
- **Responses:**
    - `200 OK`: Successful preprocessing.
    - `400 Bad Request`: Invalid preprocessing steps or data format.
    - `500 Internal Server Error`: Server-side error.

### 4. Augment Data
- **Endpoint:** `/augment/{data_type}`
- **Method:** `POST`
- **Description:** Applies augmentation techniques to the uploaded data.
- **Parameters:**
    - `data_type` (path): Type of data.
- **Request Body:** JSON with `data` and `augmentation_techniques`.
- **Responses:**
    - `200 OK`: Successful augmentation.
    - `400 Bad Request`: Invalid augmentation techniques or data format.
    - `500 Internal Server Error`: Server-side error.

## Dependencies

The project relies on the following key dependencies:

- **FastAPI:** Web framework for building APIs.
- **Uvicorn:** ASGI server for running FastAPI applications.
- **Python-Multipart:** For handling file uploads.
- **NLTK:** Natural Language Toolkit for text processing.
- **Torch:** PyTorch for machine learning operations.
- **Librosa:** Audio processing library.
- **OpenCV-Python:** Computer vision tasks.
- **Jinja2:** Templating engine for rendering HTML.
- **SoundFile:** For audio file reading and writing.

Full list of dependencies is available in the `requirements.txt` file.

## Technical Details

### File Structure

- **`main.py`**
    - Entry point of the FastAPI application.
    - Defines API endpoints for data upload, preprocessing, and augmentation.
- **`utils.py`**
    - Contains utility functions for loading and displaying data.
    - Includes functions like `load_data`, `show_sample`, and `visualize_3d_geometry`.
- **`preprocessing.py`**
    - Implements preprocessing steps for different data types.
    - Functions for text normalization, audio trimming, image resizing, etc.
- **`augmentation.py`**
    - Implements augmentation techniques for expanding datasets.
    - Includes functions like synonym replacement, time stretching, random rotation, etc.
- **`app/static/`**
    - Contains static files such as CSS and JavaScript.
- **`app/templates/index.html`**
    - Frontend interface for interacting with the application.
- **`requirements.txt`**
    - Lists all Python dependencies required for the project.

### Data Handling

- **Text:**
    - Supports plain text and common document formats.
    - Preprocessing includes padding, punctuation removal, tokenization, and embedding.
    - Augmentation includes synonym replacement and random insertion.

- **Audio:**
    - Supports WAV, MP3, and OGG formats.
    - Preprocessing includes normalization, silence trimming, resampling, and MFCC extraction.
    - Augmentation includes time stretching, pitch shifting, and adding noise.

- **Image:**
    - Supports PNG, JPEG, and GIF formats.
    - Preprocessing includes resizing, normalization, and grayscale conversion.
    - Augmentation includes rotation, flipping, and noise addition.

- **3D Geometry:**
    - Supports OFF file format.
    - Preprocessing includes normalization and centering.
    - Augmentation includes random rotation, scaling, and noise addition.
    - Visualization is provided through generated projection images.

### Error Handling

The application incorporates comprehensive error handling mechanisms:

- **Validation Errors:** Ensures uploaded files meet the required formats and specifications.
- **Processing Errors:** Catches and logs errors during data processing and augmentation.
- **Server Errors:** Handles unexpected server-side issues gracefully with appropriate logging.

### Logging

Logging is configured using Python's built-in `logging` module:

- **INFO Level:** Logs general information about received requests and processing steps.
- **DEBUG Level:** Provides detailed insights into data structures and processing flow.
- **ERROR Level:** Captures and logs errors encountered during execution.

Logs are essential for monitoring application behavior and diagnosing issues.

## Contributing

Contributions are welcome! To contribute to the **DataAlchemy**, please follow these steps:

1. **Fork the Repository**
    - Click the "Fork" button on the repository page to create your own fork.

2. **Clone Your Fork**
    ```bash
    git clone https://github.com/dhairyag/AI_Quest_Lab.git
    cd session_03_ML_data_processing_app
    ```

3. **Create a New Branch**
    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make Your Changes**
    - Implement your feature or fix.
    - Ensure your code adheres to the project's coding standards.

5. **Commit Your Changes**
    ```bash
    git commit -m "Add feature: your-feature-name"
    ```

6. **Push to Your Fork**
    ```bash
    git push origin feature/your-feature-name
    ```

7. **Create a Pull Request**
    - Navigate to your fork on GitHub.
    - Click the "Compare & pull request" button.
    - Provide a clear description of your changes and submit the pull request.

