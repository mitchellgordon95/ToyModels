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
        updateSuperpositionPlot();  // Update every 50 steps
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
    const canvases = ['weight-canvas', 'superposition-canvas', 'loss-canvas'];
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

function updateSuperpositionPlot() {
    if (!model) return;
    
    const canvas = document.getElementById('superposition-canvas');
    const ctx = canvas.getContext('2d');
    
    const W = model.getWeights();
    const m = W.length;     // hidden dim
    const n = W[0].length;  // input dim
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 250;
    
    const margin = { top: 40, right: 120, bottom: 30, left: 40 };  // Increased top and right margins
    const width = canvas.width - margin.left - margin.right;
    const height = canvas.height - margin.top - margin.bottom;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate norms and interference for each feature
    const featureData = [];
    for (let i = 0; i < n; i++) {
        // Extract column i from W (feature i's representation)
        const W_i = [];
        for (let j = 0; j < m; j++) {
            W_i.push(W[j][i]);
        }
        
        // Calculate norm
        const norm = Math.sqrt(W_i.reduce((sum, val) => sum + val * val, 0));
        
        // Calculate interference (superposition)
        let interference = 0;
        if (norm > 1e-6) {
            // Normalize W_i
            const W_i_hat = W_i.map(val => val / norm);
            
            // For each other feature j
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    // Extract column j from W
                    const W_j = [];
                    for (let k = 0; k < m; k++) {
                        W_j.push(W[k][j]);
                    }
                    
                    // Compute dot product
                    let dot = 0;
                    for (let k = 0; k < m; k++) {
                        dot += W_i_hat[k] * W_j[k];
                    }
                    
                    interference += dot * dot;
                }
            }
        }
        
        featureData.push({ index: i, norm: norm, interference: interference });
    }
    
    // Find max norm for scaling
    const maxNorm = Math.max(...featureData.map(d => d.norm));
    
    // Draw bars
    const barHeight = Math.max(1, height / n - 1);
    
    for (let i = 0; i < n; i++) {
        const data = featureData[i];
        const barWidth = (data.norm / maxNorm) * width;
        
        // Clip interference to [0, 1] for color mapping
        const colorValue = Math.min(1, data.interference);
        
        // Black to yellow gradient
        const r = Math.floor(colorValue * 255);
        const g = Math.floor(colorValue * 255);
        const b = 0;
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
            margin.left,
            margin.top + i * (height / n),
            barWidth,
            barHeight
        );
        
        // Feature index label
        if (n <= 30 || i % Math.ceil(n/30) === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(`${i}`, margin.left - 3, margin.top + i * (height / n) + barHeight/2 + 3);
        }
    }
    
    // Draw axes
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + height);
    ctx.lineTo(margin.left + width, margin.top + height);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + height);
    ctx.stroke();
    
    // X-axis labels
    ctx.fillStyle = '#666';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('0', margin.left, margin.top + height + 15);
    ctx.fillText(maxNorm.toFixed(2), margin.left + width, margin.top + height + 15);
    
    // Title
    ctx.font = '12px sans-serif';
    ctx.fillText('||W_i|| (feature magnitude)', canvas.width / 2, canvas.height - 5);
    
    // Color legend - positioned in the right margin area
    const legendX = margin.left + width + 20;
    const legendY = margin.top;
    const legendWidth = 80;
    const legendHeight = 15;
    
    // Create gradient
    const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0);
    gradient.addColorStop(0, 'rgb(0, 0, 0)');     // Black (no interference)
    gradient.addColorStop(1, 'rgb(255, 255, 0)'); // Yellow (high interference)
    
    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    
    ctx.strokeStyle = '#ccc';
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);
    
    ctx.fillStyle = '#666';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Superposition', legendX + legendWidth/2, legendY + legendHeight + 12);
    ctx.textAlign = 'left';
    ctx.fillText('0', legendX, legendY - 3);
    ctx.textAlign = 'right';
    ctx.fillText('1+', legendX + legendWidth, legendY - 3);
}

function updateFinalVisualization() {
    updateWeightMatrix();
    updateLossPlot();
    updateSuperpositionPlot();
    
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