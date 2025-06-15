let model = null;
let isTraining = false;

document.addEventListener('DOMContentLoaded', () => {
    setupControls();
    initializeModel();
    updateImportancePreview();
});

function setupControls() {
    const inputDimSlider = document.getElementById('input-dim');
    const hiddenDimSlider = document.getElementById('hidden-dim');
    const importanceSlider = document.getElementById('importance');
    
    inputDimSlider.addEventListener('input', (e) => {
        document.getElementById('input-dim-display').textContent = e.target.value;
        const hiddenMax = Math.max(1, parseInt(e.target.value));
        hiddenDimSlider.max = hiddenMax;
        if (parseInt(hiddenDimSlider.value) > hiddenMax) {
            hiddenDimSlider.value = hiddenMax;
            document.getElementById('hidden-dim-display').textContent = hiddenMax;
        }
        updateImportancePreview();
    });
    
    hiddenDimSlider.addEventListener('input', (e) => {
        document.getElementById('hidden-dim-display').textContent = e.target.value;
    });
    
    importanceSlider.addEventListener('input', (e) => {
        document.getElementById('importance-display').textContent = parseFloat(e.target.value).toFixed(2);
        updateImportancePreview();
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
    
    // Reinitialize the model with fresh weights
    initializeModel();
    
    isTraining = true;
    document.getElementById('train-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    
    disableParameterControls(true);
    
    const sparsity = parseFloat(document.getElementById('sparsity').value);
    const importance = parseFloat(document.getElementById('importance').value);
    
    // Get values from More Settings
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const steps = parseInt(document.getElementById('steps').value);
    const lrSchedule = document.getElementById('lr-schedule').value;
    
    updateStatus('Training...');
    
    const trainingParams = {
        steps: steps,
        batchSize: batchSize,
        learningRate: learningRate,
        lrSchedule: lrSchedule,
        sparsity: sparsity,
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

function updateTrainingProgress({ step, loss, learningRate, converged }) {
    document.getElementById('step').textContent = step;
    document.getElementById('loss').textContent = loss.toExponential(3);
    
    // Update status with learning rate info
    if (learningRate) {
        updateStatus(`Training... (LR: ${learningRate.toExponential(2)})`);
    }
    
    if (step % 50 === 0) {
        updateLossPlot();
        updateWeightMatrix();
        if (step % 200 === 0) {  // Less frequent for performance
            updateReconstructionQuality();
        }
    }
}

function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

function disableParameterControls(disabled) {
    const controls = [
        'input-dim', 'hidden-dim', 'linear-mode', 'relu-mode',
        'learning-rate', 'batch-size', 'steps', 'lr-schedule'
    ];
    
    controls.forEach(id => {
        document.getElementById(id).disabled = disabled;
    });
}

function clearVisualizations() {
    const canvases = ['weight-canvas', 'reconstruction-canvas', 'loss-canvas'];
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
    ctx.fillText(`Steps: ${losses.length}`, width - 100, 15);
}

function updateWeightMatrix() {
    if (!model) return;
    
    const canvas = document.getElementById('weight-canvas');
    const ctx = canvas.getContext('2d');
    
    // Get W^T W (Gram matrix) and bias
    const analysis = model.analyzeRepresentation();
    const gramMatrix = analysis.gramMatrix;
    const bias = model.bias;
    const n = gramMatrix.length;
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 250;
    
    // Calculate dimensions for both visualizations
    const totalWidth = canvas.width;
    const matrixWidth = Math.floor(totalWidth * 0.6); // 60% for matrix
    const biasWidth = Math.floor(totalWidth * 0.3); // 30% for bias
    const gap = 20; // gap between visualizations
    
    // Matrix dimensions
    const legendWidth = 50;
    const matrixPlotArea = matrixWidth - legendWidth - 20;
    const cellSize = Math.min(15, matrixPlotArea / n, (canvas.height - 40) / n);
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
    
    // Legend gradient for W^T W
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
    
    // Title for matrix
    ctx.font = '12px sans-serif';
    ctx.fillText(`W^T W (${n}×${n})`, startX, canvas.height - 18);
    ctx.fillText(`Orthogonality: ${analysis.orthogonality.toFixed(3)}`, startX, canvas.height - 5);
    
    // Draw bias visualization
    const biasStartX = matrixWidth + gap;
    const biasPlotWidth = biasWidth - 80; // Leave space for scale
    
    // Find max bias value for scaling
    let maxBias = 0;
    for (let i = 0; i < n; i++) {
        maxBias = Math.max(maxBias, Math.abs(bias[i]));
    }
    maxBias = Math.max(maxBias, 0.01); // Ensure minimum scale
    
    // Draw bias heatmap
    for (let i = 0; i < n; i++) {
        const biasValue = bias[i];
        const normalizedBias = biasValue / maxBias;
        
        // Color gradient from orange (negative) to white (zero) to green (positive)
        let r, g, b;
        if (normalizedBias > 0) {
            // Positive: white to green
            r = 255 - Math.floor(normalizedBias * 155);
            g = 255 - Math.floor(normalizedBias * 55);
            b = 255 - Math.floor(normalizedBias * 155);
        } else {
            // Negative: white to orange
            r = 255;
            g = 255 + Math.floor(normalizedBias * 90);
            b = 255 + Math.floor(normalizedBias * 180);
        }
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
            biasStartX,
            startY + i * cellSize,
            biasPlotWidth,
            cellSize - 1
        );
    }
    
    // Bias color scale
    const biasScaleX = biasStartX + biasPlotWidth + 10;
    const biasScaleWidth = 20;
    const biasScaleHeight = Math.min(200, n * cellSize);
    
    // Create gradient for bias
    const biasGradient = ctx.createLinearGradient(0, startY, 0, startY + biasScaleHeight);
    biasGradient.addColorStop(0, 'rgb(100, 200, 100)');     // Green (positive)
    biasGradient.addColorStop(0.5, 'rgb(255, 255, 255)');   // White (zero)
    biasGradient.addColorStop(1, 'rgb(255, 165, 75)');      // Orange (negative)
    
    ctx.fillStyle = biasGradient;
    ctx.fillRect(biasScaleX, startY, biasScaleWidth, biasScaleHeight);
    
    // Bias scale border
    ctx.strokeStyle = '#ccc';
    ctx.strokeRect(biasScaleX, startY, biasScaleWidth, biasScaleHeight);
    
    // Bias scale labels
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`+${maxBias.toFixed(3)}`, biasScaleX + biasScaleWidth + 5, startY + 5);
    ctx.fillText('0', biasScaleX + biasScaleWidth + 5, startY + biasScaleHeight/2 + 3);
    ctx.fillText(`-${maxBias.toFixed(3)}`, biasScaleX + biasScaleWidth + 5, startY + biasScaleHeight);
    
    // Bias title
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Bias', biasStartX + biasPlotWidth/2, canvas.height - 5);
}

function updateReconstructionQuality() {
    if (!model) return;
    
    const canvas = document.getElementById('reconstruction-canvas');
    const ctx = canvas.getContext('2d');
    
    const sparsity = parseFloat(document.getElementById('sparsity').value);
    const importanceDecay = parseFloat(document.getElementById('importance').value);
    
    // Compute reconstruction quality for each feature
    const { qualities, importanceVector, featureCounts } = 
        model.computeFeatureReconstructionQuality(500, sparsity, importanceDecay);
    
    const n = qualities.length;
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 250;
    
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const width = canvas.width - margin.left - margin.right;
    const height = canvas.height - margin.top - margin.bottom;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Create feature indices sorted by importance
    const indices = Array.from({length: n}, (_, i) => i);
    indices.sort((a, b) => importanceVector[b] - importanceVector[a]);
    
    // Draw bars
    const barWidth = Math.max(1, width / n - 1);
    const maxQuality = 1;
    
    for (let i = 0; i < n; i++) {
        const featureIdx = indices[i];
        const quality = qualities[featureIdx];
        const importance = importanceVector[featureIdx];
        
        // Bar height
        const barHeight = quality * height;
        
        // Color based on importance (gradient from blue to light gray)
        const colorIntensity = importance;
        const r = Math.floor(200 - colorIntensity * 150);
        const g = Math.floor(200 - colorIntensity * 100);
        const b = Math.floor(200 + colorIntensity * 55);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
            margin.left + i * (width / n),
            margin.top + height - barHeight,
            barWidth,
            barHeight
        );
    }
    
    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + height);
    ctx.stroke();
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + height);
    ctx.lineTo(margin.left + width, margin.top + height);
    ctx.stroke();
    
    // Y-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', margin.left - 5, margin.top + 5);
    ctx.fillText('0.5', margin.left - 5, margin.top + height/2 + 5);
    ctx.fillText('0.0', margin.left - 5, margin.top + height + 5);
    
    // Title and labels
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Features (sorted by importance)', canvas.width / 2, canvas.height - 10);
    
    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Reconstruction Quality', 0, 0);
    ctx.restore();
    
    // Legend
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgb(50, 100, 255)';
    ctx.fillRect(canvas.width - 100, 10, 15, 10);
    ctx.fillStyle = '#666';
    ctx.fillText('High importance', canvas.width - 80, 19);
    
    ctx.fillStyle = 'rgb(200, 200, 200)';
    ctx.fillRect(canvas.width - 100, 25, 15, 10);
    ctx.fillStyle = '#666';
    ctx.fillText('Low importance', canvas.width - 80, 34);
}

function updateFinalVisualization() {
    updateWeightMatrix();
    updateLossPlot();
    updateReconstructionQuality();
    
    const analysis = model.analyzeRepresentation();
    console.log('Model analysis:', analysis);
}

function updateImportancePreview() {
    const canvas = document.getElementById('importance-preview-canvas');
    const ctx = canvas.getContext('2d');
    const inputDim = parseInt(document.getElementById('input-dim').value);
    const importanceDecay = parseFloat(document.getElementById('importance').value);
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 80;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get the same importance vector used in the model
    const importanceVector = SuperpositionModel.computeImportanceVector(inputDim, importanceDecay);
    
    // Calculate bar width and spacing
    const margin = 10;
    const maxBars = Math.min(inputDim, 50); // Limit display to 50 bars for readability
    const barWidth = Math.max(1, (canvas.width - 2 * margin) / maxBars - 2);
    const spacing = Math.max(1, (canvas.width - 2 * margin) / maxBars);
    
    // Draw bars
    for (let i = 0; i < maxBars; i++) {
        const importance = importanceVector[i];
        const barHeight = importance * (canvas.height - 20);
        
        ctx.fillStyle = `hsl(${210 - i * 3}, 70%, 50%)`;
        ctx.fillRect(
            margin + i * spacing,
            canvas.height - 10 - barHeight,
            barWidth,
            barHeight
        );
    }
    
    // Draw labels
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    
    // Label first and last feature
    ctx.fillText('1', margin + barWidth/2, canvas.height - 1);
    if (maxBars > 1) {
        ctx.fillText(maxBars.toString(), margin + (maxBars - 1) * spacing + barWidth/2, canvas.height - 1);
    }
    
    // Show continuation indicator if needed
    if (inputDim > maxBars) {
        ctx.fillText(`... (${inputDim} total)`, canvas.width - 40, canvas.height - 1);
    }
}