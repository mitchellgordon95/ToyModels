let model = null;
let isTraining = false;

document.addEventListener('DOMContentLoaded', () => {
    setupControls();
    initializeModel();
});

function setupControls() {
    const inputDimSlider = document.getElementById('input-dim');
    const hiddenDimSlider = document.getElementById('hidden-dim');
    const sparsitySlider = document.getElementById('sparsity');
    const importanceSlider = document.getElementById('importance');
    
    inputDimSlider.addEventListener('input', (e) => {
        document.getElementById('input-dim-display').textContent = e.target.value;
        const hiddenMax = Math.max(1, parseInt(e.target.value) - 1);
        hiddenDimSlider.max = hiddenMax;
        if (parseInt(hiddenDimSlider.value) > hiddenMax) {
            hiddenDimSlider.value = hiddenMax;
            document.getElementById('hidden-dim-display').textContent = hiddenMax;
        }
    });
    
    hiddenDimSlider.addEventListener('input', (e) => {
        document.getElementById('hidden-dim-display').textContent = e.target.value;
    });
    
    sparsitySlider.addEventListener('input', (e) => {
        document.getElementById('sparsity-display').textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    importanceSlider.addEventListener('input', (e) => {
        document.getElementById('importance-display').textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    document.getElementById('train-btn').addEventListener('click', startTraining);
    document.getElementById('stop-btn').addEventListener('click', stopTraining);
    
    document.querySelectorAll('input[name="activation"]').forEach(radio => {
        radio.addEventListener('change', initializeModel);
    });
    
    [inputDimSlider, hiddenDimSlider].forEach(el => {
        el.addEventListener('change', initializeModel);
    });
}

function initializeModel() {
    const inputDim = parseInt(document.getElementById('input-dim').value);
    const hiddenDim = parseInt(document.getElementById('hidden-dim').value);
    const activation = document.querySelector('input[name="activation"]:checked').value;
    
    model = new SuperpositionModel(inputDim, hiddenDim, activation);
    
    updateStatus('Model initialized');
    clearVisualizations();
}

async function startTraining() {
    if (isTraining) return;
    
    isTraining = true;
    document.getElementById('train-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    
    disableParameterControls(true);
    
    const sparsity = parseFloat(document.getElementById('sparsity').value);
    const importance = parseFloat(document.getElementById('importance').value);
    
    updateStatus('Training...');
    
    const trainingParams = {
        epochs: 5000,
        batchSize: 32,
        initialLearningRate: 0.1,
        minLearningRate: 0.001,
        decayRate: 0.995,
        sparsity: sparsity,
        sparsityWeight: 0.1,
        importance: importance,
        convergenceThreshold: 1e-5,
        callback: updateTrainingProgress
    };
    
    try {
        const result = await model.train(trainingParams);
        
        if (result.converged) {
            updateStatus('Training converged!');
        } else {
            updateStatus('Training completed');
        }
        
        updateFinalVisualization();
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Training error occurred');
    }
    
    isTraining = false;
    document.getElementById('train-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    disableParameterControls(false);
}

function stopTraining() {
    if (model) {
        model.stop();
    }
    isTraining = false;
    document.getElementById('train-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    disableParameterControls(false);
    updateStatus('Training stopped');
}

function updateTrainingProgress({ epoch, loss, learningRate, converged }) {
    document.getElementById('epoch').textContent = epoch;
    document.getElementById('loss').textContent = loss.toExponential(3);
    
    // Update status with learning rate info
    if (learningRate) {
        updateStatus(`Training... (LR: ${learningRate.toExponential(2)})`);
    }
    
    if (epoch % 50 === 0) {
        updateLossPlot();
        updateWeightMatrix();
    }
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function disableParameterControls(disabled) {
    const controls = [
        'input-dim', 'hidden-dim', 'linear-mode', 'relu-mode'
    ];
    
    controls.forEach(id => {
        document.getElementById(id).disabled = disabled;
    });
}

function clearVisualizations() {
    const canvases = ['weight-canvas', 'reconstruction-canvas', 'feature-canvas', 'loss-canvas'];
    canvases.forEach(id => {
        const canvas = document.getElementById(id);
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
}

function updateLossPlot() {
    if (!model || model.lossHistory.length === 0) return;
    
    const canvas = document.getElementById('loss-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 250;
    
    ctx.clearRect(0, 0, width, height);
    
    const losses = model.lossHistory;
    const maxLoss = Math.max(...losses);
    const minLoss = Math.min(...losses);
    const range = maxLoss - minLoss || 1;
    
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    losses.forEach((loss, i) => {
        const x = (i / (losses.length - 1)) * (width - 40) + 20;
        const y = height - 30 - ((loss - minLoss) / range) * (height - 60);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${minLoss.toExponential(2)} - ${maxLoss.toExponential(2)}`, 10, 15);
    ctx.fillText(`Epochs: ${losses.length}`, width - 100, 15);
}

function updateWeightMatrix() {
    if (!model) return;
    
    const canvas = document.getElementById('weight-canvas');
    const ctx = canvas.getContext('2d');
    
    // Get W^T W (Gram matrix)
    const analysis = model.analyzeRepresentation();
    const gramMatrix = analysis.gramMatrix;
    const n = gramMatrix.length;
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 250;
    
    // Calculate cell size and plotting area
    const legendWidth = 60;
    const plotArea = canvas.width - legendWidth - 30;
    const cellSize = Math.min(20, plotArea / n, (canvas.height - 40) / n);
    const startX = 15;
    const startY = 20;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Find max absolute value for scaling
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            maxVal = Math.max(maxVal, Math.abs(gramMatrix[i][j]));
        }
    }
    maxVal = Math.max(maxVal, 1); // Ensure maxVal is at least 1
    
    // Draw the matrix
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const value = gramMatrix[i][j];
            const normalizedValue = value / maxVal;
            
            // Use a diverging color scheme
            let r, g, b;
            if (normalizedValue > 0) {
                // Positive values: white to blue
                r = 255 - Math.floor(normalizedValue * 203);
                g = 255 - Math.floor(normalizedValue * 103);
                b = 255;
            } else {
                // Negative values: white to red
                r = 255;
                g = 255 + Math.floor(normalizedValue * 179);
                b = 255 + Math.floor(normalizedValue * 195);
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(
                startX + j * cellSize,
                startY + i * cellSize,
                cellSize - 1,
                cellSize - 1
            );
        }
    }
    
    // Draw color legend
    const legendX = startX + n * cellSize + 20;
    const legendHeight = Math.min(200, n * cellSize);
    const legendY = startY;
    
    // Legend gradient
    const gradient = ctx.createLinearGradient(0, legendY, 0, legendY + legendHeight);
    gradient.addColorStop(0, 'rgb(52, 152, 255)');     // Blue (positive)
    gradient.addColorStop(0.5, 'rgb(255, 255, 255)'); // White (zero)
    gradient.addColorStop(1, 'rgb(255, 76, 60)');      // Red (negative)
    
    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY, 20, legendHeight);
    
    // Legend border
    ctx.strokeStyle = '#ccc';
    ctx.strokeRect(legendX, legendY, 20, legendHeight);
    
    // Legend labels
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${maxVal.toFixed(2)}`, legendX + 25, legendY + 5);
    ctx.fillText('0.00', legendX + 25, legendY + legendHeight / 2 + 3);
    ctx.fillText(`-${maxVal.toFixed(2)}`, legendX + 25, legendY + legendHeight);
    
    // Title
    ctx.font = '12px sans-serif';
    ctx.fillText(`W^T W Gram Matrix (${n}×${n})`, startX, canvas.height - 5);
    ctx.fillText(`Orthogonality: ${analysis.orthogonality.toFixed(3)}`, startX + 150, canvas.height - 5);
}

function updateFinalVisualization() {
    updateWeightMatrix();
    updateLossPlot();
    
    const analysis = model.analyzeRepresentation();
    console.log('Model analysis:', analysis);
}