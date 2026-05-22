// API endpoints
const API_RECOGNIZE = '/api/recognize';

// DOM elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const previewButtons = document.querySelector('.preview-buttons');
const detectUploadBtn = document.getElementById('detect-upload-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results-section');
const resultsSummary = document.getElementById('results-summary');
const resultsImage = document.getElementById('results-image');
const detectionsList = document.getElementById('detections-list');
const errorMessage = document.getElementById('error-message');

let selectedFile = null;

// Upload area click
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        previewButtons.style.display = 'flex';
        uploadArea.style.display = 'none'; // Hide upload area
        hideResults();
        hideError();
    };
    reader.readAsDataURL(file);
}

// Recognize faces
detectUploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    setButtonLoading(detectUploadBtn, true);
    hideError();
    hideResults();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(API_RECOGNIZE, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Recognition failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        showError(error.message);
    } finally {
        setButtonLoading(detectUploadBtn, false);
    }
});

// Clear button - reset everything
clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    previewContainer.style.display = 'none';
    previewButtons.style.display = 'none';
    previewImage.src = '';
    uploadArea.style.display = 'block'; // Show upload area again
    hideResults();
    hideError();
});

// Display results
function displayResults(result) {
    if (!result.success) {
        showError(result.message || 'Recognition failed');
        return;
    }

    resultsSection.style.display = 'block';
    resultsSummary.textContent = result.message || `Found ${result.count} face(s)`;
    resultsImage.src = `data:image/jpeg;base64,${result.image}`;

    detectionsList.innerHTML = '';

    result.detections.forEach((detection, index) => {
        const item = createDetectionItem(detection, index);
        detectionsList.appendChild(item);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Create detection item
function createDetectionItem(detection, index) {
    const item = document.createElement('div');
    item.className = 'detection-item';

    // Add access status class
    if (detection.access === 'granted') {
        item.classList.add('access-granted');
    } else {
        item.classList.add('access-denied');
    }

    const name = document.createElement('div');
    name.className = 'detection-class';
    name.textContent = detection.name;

    const status = document.createElement('div');
    status.className = 'detection-status';
    status.textContent = detection.access === 'granted' ? '✅ ACCESS GRANTED' : '❌ ACCESS DENIED';

    if (detection.confidence > 0) {
        const confidence = document.createElement('div');
        confidence.className = 'detection-confidence';
        confidence.textContent = `Confidence: ${(detection.confidence * 100).toFixed(1)}%`;

        const bar = document.createElement('div');
        bar.className = 'confidence-bar';
        const fill = document.createElement('div');
        fill.className = 'confidence-fill';
        fill.style.width = `${detection.confidence * 100}%`;
        bar.appendChild(fill);

        item.appendChild(name);
        item.appendChild(status);
        item.appendChild(confidence);
        item.appendChild(bar);
    } else {
        item.appendChild(name);
        item.appendChild(status);
    }

    return item;
}

// Helper functions
function setButtonLoading(button, loading) {
    if (loading) {
        button.disabled = true;
        button.innerHTML = '<span class="btn-loader"></span> Processing...';
    } else {
        button.disabled = false;
        button.innerHTML = '🔍 Recognize Faces';
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}
