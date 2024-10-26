let currentData = null;

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const dataType = document.getElementById('data-type').value;
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch(`/upload/${dataType}`, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        currentData = result.sample;
        document.getElementById('results').textContent = JSON.stringify(result, null, 2);
        updateProcessingOptions(dataType);
    } catch (error) {
        console.error('Error:', error);
    }
});

function updateProcessingOptions(dataType) {
    const preprocessingOptions = document.getElementById('preprocessing-options');
    const augmentationOptions = document.getElementById('augmentation-options');
    
    preprocessingOptions.innerHTML = '';
    augmentationOptions.innerHTML = '';

    if (dataType === 'text') {
        ['tokenize', 'pad', 'embed'].forEach(option => {
            preprocessingOptions.innerHTML += `<label><input type="checkbox" name="preprocess" value="${option}"> ${option}</label>`;
        });
        ['synonym_replacement', 'random_insertion'].forEach(option => {
            augmentationOptions.innerHTML += `<label><input type="checkbox" name="augment" value="${option}"> ${option}</label>`;
        });
    }
    // Add options for other data types here
}

document.getElementById('preprocess-btn').addEventListener('click', async () => {
    if (!currentData) return;
    const dataType = document.getElementById('data-type').value;
    const steps = Array.from(document.querySelectorAll('input[name="preprocess"]:checked')).map(el => el.value);
    
    try {
        const response = await fetch(`/preprocess/${dataType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({data: currentData, preprocessing_steps: steps})
        });
        const result = await response.json();
        currentData = result.sample;
        document.getElementById('results').textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        console.error('Error:', error);
    }
});

document.getElementById('augment-btn').addEventListener('click', async () => {
    if (!currentData) return;
    const dataType = document.getElementById('data-type').value;
    const techniques = Array.from(document.querySelectorAll('input[name="augment"]:checked')).map(el => el.value);
    
    try {
        const response = await fetch(`/augment/${dataType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({data: currentData, augmentation_techniques: techniques})
        });
        const result = await response.json();
        currentData = result.sample;
        document.getElementById('results').textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        console.error('Error:', error);
    }
});
