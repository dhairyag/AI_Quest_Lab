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
        
        // Display the original sample
        displaySample(result.sample.original, result.data_type, 'originalData');
        
        // Store the current data
        currentData = result.sample.original;
        
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

function displaySample(sample, dataType, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (dataType === 'audio') {
        console.log('Received audio sample:', sample);

        const audioContainer = document.createElement('div');
        audioContainer.className = 'audio-controls';
        
        // Create audio element
        const audio = new Audio();
        
        // Add error handling before setting source
        audio.onerror = function(e) {
            console.error('Audio error:', e);
            console.error('Audio error code:', audio.error?.code);
            console.error('Audio error message:', audio.error?.message);
            
            // Show error message to user
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'Error loading audio. Please check the file format.';
            audioContainer.appendChild(errorDiv);
        };
        
        // Validate audio data
        if (!sample.audio_data) {
            console.error('No audio data received');
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'No audio data received';
            audioContainer.appendChild(errorDiv);
            container.appendChild(audioContainer);
            return;
        }

        // Set audio source
        const audioSrc = `data:audio/wav;base64,${sample.audio_data}`;
        audio.src = audioSrc;
        audio.controls = true;
        
        // Debug log
        console.log('Setting audio source, length:', sample.audio_data.length);
        
        // Create play/pause button
        const playPauseBtn = document.createElement('button');
        playPauseBtn.className = 'play-pause-btn';
        playPauseBtn.innerHTML = '▶️';
        playPauseBtn.disabled = true;
        
        // Enable button when audio is ready
        audio.addEventListener('canplaythrough', () => {
            console.log('Audio can play through');
            playPauseBtn.disabled = false;
        });
        
        playPauseBtn.onclick = function() {
            if (audio.paused) {
                audio.play()
                .then(() => {
                    console.log('Audio playing');
                    playPauseBtn.innerHTML = '⏸️';
                })
                .catch(error => {
                    console.error('Play error:', error);
                });
            } else {
                audio.pause();
                playPauseBtn.innerHTML = '▶️';
            }
        };
        
        // Add audio element first
        audioContainer.appendChild(audio);
        
        // Add play/pause button
        audioContainer.appendChild(playPauseBtn);
        
        // Add duration display
        const timeDisplay = document.createElement('div');
        timeDisplay.className = 'time-display';
        audioContainer.appendChild(timeDisplay);
        
        // Update time display
        audio.ontimeupdate = function() {
            const currentTime = Math.floor(audio.currentTime);
            const duration = Math.floor(audio.duration);
            timeDisplay.textContent = `${currentTime}s / ${duration}s`;
        };
        
        // Add sample rate display
        if (sample.sample_rate) {
            const sampleRateDiv = document.createElement('div');
            sampleRateDiv.className = 'sample-rate';
            sampleRateDiv.textContent = `Sample Rate: ${sample.sample_rate} Hz`;
            audioContainer.appendChild(sampleRateDiv);
        }
        
        container.appendChild(audioContainer);
    } else if (dataType === 'text') {
        // Existing text handling...
    } else if (dataType === 'image') {
        // Existing image handling...
    }
}
