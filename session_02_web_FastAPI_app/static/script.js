function selectIcon(animal) {
    console.log(`Selecting icon: ${animal}`);
    const resultBox = document.getElementById('resultBox');
    const imagePath = `/static/images/${animal}.jpg`;  // or .png if your images are PNGs
    console.log(`Image path: ${imagePath}`);
    // print the image path
    console.log(imagePath);
    
    // Create a new image element
    const img = new Image();
    img.onload = function() {
        console.log('Image loaded successfully');
        resultBox.innerHTML = '';
        resultBox.appendChild(img);
    };
    img.onerror = function() {
        console.error(`Failed to load image: ${imagePath}`);
        resultBox.innerHTML = `<p>Error loading image for ${animal}</p>`;
    };
    img.src = imagePath;
    img.alt = animal;
    img.style.width = '100%';

    console.log('Image loading initiated');
}

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const resultBox = document.getElementById('resultBox');
            resultBox.innerHTML = `
                <p>File Name: ${data.filename}</p>
                <p>File Size: ${data.size} bytes</p>
                <p>File Type: ${data.type}</p>
            `;
        })
        .catch(error => console.error('Error:', error));
    }
}
