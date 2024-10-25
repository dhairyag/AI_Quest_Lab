# Image and File Processor

## Project Overview

This project is a web application that allows users to select animal icons and upload files. It uses FastAPI as the backend and HTML/CSS/JavaScript for the frontend.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Running the Application](#running-the-application)
3. [Technical Details](#technical-details)
4. [API Endpoints](#api-endpoints)
5. [Frontend Components](#frontend-components)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)


## Setup and Installation

1. Ensure you have Python 3.7+ installed.
2. Clone this repository:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
5. Install the required packages:
   ```
   pip install fastapi uvicorn python-multipart jinja2
   ```
   Alternatively, if you have a `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure you're in the project root directory and your virtual environment is activated.
2. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```
3. Open a web browser and navigate to `http://127.0.0.1:8000`

## Technical Details

### Backend

- **Framework**: FastAPI
- **Python Version**: 3.7+
- **ASGI Server**: Uvicorn
- **Additional Dependencies**:
  - `python-multipart`: For handling file uploads
  - `jinja2`: For templating

### Frontend

- **HTML5**
- **CSS3**
- **JavaScript (ES6+)**

### Static File Serving

Static files are served from the `static/` directory. This includes CSS, JavaScript, and image files.

### Templating

The application uses Jinja2 templates, with the main template located at `templates/index.html`.

## API Endpoints

- `GET /`: Serves the main HTML page
- `POST /upload`: Handles file uploads
  - Request body: `multipart/form-data` with a `file` field
  - Response: JSON object with file details (name, size, type)

## Frontend Components

### HTML Structure

The `index.html` file contains the basic structure of the web page, including:
- A left box with animal icons and a file upload area
- A right box for displaying results

### CSS Styling

The `style.css` file contains all the styles for the application, including layout, colors, and responsive design.

### JavaScript Functionality

The `script.js` file contains two main functions:
- `selectIcon(animal)`: Displays the selected animal image
- `uploadFile()`: Handles file uploads and displays file information

## Troubleshooting

- If images are not displaying, ensure all image files are present in the `static/images/` directory and that file names match those in the HTML and JavaScript code.
- For any server errors, check the console where you're running `uvicorn` for error messages.
- For frontend issues, use the browser's developer tools to check for console errors or network request issues.

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Create a pull request

Please ensure your code adheres to the existing style conventions.
