class SuperpositionModel {
    constructor(inputDim, hiddenDim, activation = 'linear') {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.activation = activation;
        
        // Initialize with Xavier/Glorot initialization
        const scale = Math.sqrt(2.0 / (inputDim + hiddenDim));
        this.W = Matrix.randomNormal(hiddenDim, inputDim, 0, scale);
        
        // Initialize bias term (b_final)
        this.bias = new Array(inputDim).fill(0);
        
        // Initialize AdamW optimizer state
        this.adamW = {
            mW: Matrix.zeros(hiddenDim, inputDim),
            vW: Matrix.zeros(hiddenDim, inputDim),
            mBias: new Array(inputDim).fill(0),
            vBias: new Array(inputDim).fill(0),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0
        };
        
        this.lossHistory = [];
        this.isTraining = false;
        this.totalSteps = 0;
    }
    
    static computeImportanceVector(inputDim, importanceDecay) {
        const importances = new Array(inputDim);
        for (let i = 0; i < inputDim; i++) {
            importances[i] = Math.pow(importanceDecay, i);
        }
        return importances;
    }
    
    forward(x) {
        // Encoder: x -> W -> h (no ReLU on hidden layer)
        const h = Matrix.multiplyVector(this.W, x);
        
        // Decoder: h -> W^T -> x_reconstructed
        const W_T = Matrix.transpose(this.W);
        let x_reconstructed = Matrix.multiplyVector(W_T, h);
        
        // Add bias term
        x_reconstructed = Vector.add(x_reconstructed, this.bias);
        
        // Apply ReLU to output (matching the paper)
        x_reconstructed = Vector.relu(x_reconstructed);
        
        return { h, x_reconstructed };
    }
    
    computeLoss(x, x_reconstructed, h, importanceVector) {
        // Compute importance-weighted MSE, matching the paper
        let weightedSquaredError = 0;
        for (let i = 0; i < x.length; i++) {
            const error = x[i] - x_reconstructed[i];
            weightedSquaredError += importanceVector[i] * error * error;
        }
        const totalLoss = weightedSquaredError / x.length;
        
        return {
            total: totalLoss,
            reconstruction: totalLoss,
            sparsity: 0
        };
    }
    
    backward(x, learningRate = 0.01, importanceVector) {
        const { h, x_reconstructed } = this.forward(x);
        
        const loss = this.computeLoss(x, x_reconstructed, h, importanceVector);
        
        // Gradient through ReLU on output
        const relu_grad = x_reconstructed.map(val => val > 0 ? 1 : 0);
        
        // Importance-weighted gradient
        const reconstruction_grad = [];
        for (let i = 0; i < x.length; i++) {
            reconstruction_grad[i] = (2 * importanceVector[i] * (x_reconstructed[i] - x[i]) * relu_grad[i]) / x.length;
        }
        
        // Update bias gradient
        const bias_grad = reconstruction_grad;
        this.bias = Vector.subtract(this.bias, Vector.scale(bias_grad, learningRate));
        
        const W_T = Matrix.transpose(this.W);
        let dL_dh = Matrix.multiplyVector(W_T, reconstruction_grad);
        
        const dL_dW = Matrix.zeros(this.hiddenDim, this.inputDim);
        for (let i = 0; i < this.hiddenDim; i++) {
            for (let j = 0; j < this.inputDim; j++) {
                dL_dW[i][j] = dL_dh[i] * x[j] + reconstruction_grad[j] * h[i];
            }
        }
        
        // AdamW update for weights
        this.adamW.t++;
        const t = this.adamW.t;
        const beta1 = this.adamW.beta1;
        const beta2 = this.adamW.beta2;
        const epsilon = this.adamW.epsilon;
        const weightDecay = 0.0001;
        
        // Update momentum and variance for W
        for (let i = 0; i < this.hiddenDim; i++) {
            for (let j = 0; j < this.inputDim; j++) {
                this.adamW.mW[i][j] = beta1 * this.adamW.mW[i][j] + (1 - beta1) * dL_dW[i][j];
                this.adamW.vW[i][j] = beta2 * this.adamW.vW[i][j] + (1 - beta2) * dL_dW[i][j] * dL_dW[i][j];
                
                // Bias correction
                const mHat = this.adamW.mW[i][j] / (1 - Math.pow(beta1, t));
                const vHat = this.adamW.vW[i][j] / (1 - Math.pow(beta2, t));
                
                // AdamW update with weight decay
                this.W[i][j] = this.W[i][j] - learningRate * (mHat / (Math.sqrt(vHat) + epsilon) + weightDecay * this.W[i][j]);
            }
        }
        
        // Update momentum and variance for bias
        for (let i = 0; i < this.inputDim; i++) {
            this.adamW.mBias[i] = beta1 * this.adamW.mBias[i] + (1 - beta1) * bias_grad[i];
            this.adamW.vBias[i] = beta2 * this.adamW.vBias[i] + (1 - beta2) * bias_grad[i] * bias_grad[i];
            
            // Bias correction
            const mHat = this.adamW.mBias[i] / (1 - Math.pow(beta1, t));
            const vHat = this.adamW.vBias[i] / (1 - Math.pow(beta2, t));
            
            // Adam update for bias (no weight decay on bias)
            this.bias[i] = this.bias[i] - learningRate * (mHat / (Math.sqrt(vHat) + epsilon));
        }
        
        return loss;
    }
    
    generateBatch(batchSize, sparsity, importanceDecay) {
        const batch = [];
        
        for (let i = 0; i < batchSize; i++) {
            const x = Vector.sparse(this.inputDim, sparsity);
            
            // Don't multiply by importance - that goes in the loss
            const norm = Vector.norm(x);
            if (norm > 0) {
                for (let j = 0; j < x.length; j++) {
                    x[j] /= norm;
                }
            }
            
            batch.push(x);
        }
        
        return batch;
    }
    
    trainStep(batch, learningRate, importanceVector) {
        let totalLoss = 0;
        
        for (const x of batch) {
            const loss = this.backward(x, learningRate, importanceVector);
            totalLoss += loss.total;
        }
        
        this.totalSteps++;
        
        return totalLoss / batch.length;
    }
    
    async train(params = {}) {
        const {
            steps = 10000,
            batchSize = 1024,
            learningRate = 1e-3,
            lrSchedule = 'constant',  // 'constant', 'linear', or 'cosine'
            sparsity = 0.1,
            importance = 1.0,
            convergenceThreshold = 1e-5,
            callback = null
        } = params;
        
        this.isTraining = true;
        this.lossHistory = [];
        this.totalSteps = 0;
        
        let previousLoss = Infinity;
        let convergenceCount = 0;
        let currentLearningRate = learningRate;
        
        for (let step = 0; step < steps && this.isTraining; step++) {
            // Learning rate scheduling
            if (lrSchedule === 'linear') {
                currentLearningRate = learningRate * (1 - step / steps);
            } else if (lrSchedule === 'cosine') {
                currentLearningRate = learningRate * Math.cos(0.5 * Math.PI * step / (steps - 1));
            } else {
                currentLearningRate = learningRate;  // constant
            }
            
            const batch = this.generateBatch(batchSize, sparsity, importance);
            const importanceVector = SuperpositionModel.computeImportanceVector(this.inputDim, importance);
            const loss = this.trainStep(batch, currentLearningRate, importanceVector);
            
            this.lossHistory.push(loss);
            
            // Check for convergence
            const relativeLossChange = Math.abs((loss - previousLoss) / (previousLoss + 1e-8));
            if (relativeLossChange < convergenceThreshold) {
                convergenceCount++;
                if (convergenceCount > 20) {
                    console.log(`Converged at step ${step}`);
                    break;
                }
            } else {
                convergenceCount = 0;
            }
            
            previousLoss = loss;
            
            if (callback && step % 10 === 0) {
                callback({
                    step,
                    loss,
                    learningRate: currentLearningRate,
                    converged: convergenceCount > 20
                });
                
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        this.isTraining = false;
        
        return {
            finalLoss: previousLoss,
            steps: this.lossHistory.length,
            converged: convergenceCount > 20
        };
    }
    
    stop() {
        this.isTraining = false;
    }
    
    getWeights() {
        return this.W;
    }
    
    analyzeRepresentation() {
        const W_T = Matrix.transpose(this.W);
        const gramMatrix = Matrix.multiply(W_T, this.W);
        
        let diagonalNorm = 0;
        let offDiagonalNorm = 0;
        
        for (let i = 0; i < this.inputDim; i++) {
            for (let j = 0; j < this.inputDim; j++) {
                if (i === j) {
                    diagonalNorm += gramMatrix[i][j] * gramMatrix[i][j];
                } else {
                    offDiagonalNorm += gramMatrix[i][j] * gramMatrix[i][j];
                }
            }
        }
        
        return {
            gramMatrix,
            orthogonality: Math.sqrt(diagonalNorm) / (Math.sqrt(diagonalNorm + offDiagonalNorm) + 1e-8)
        };
    }
    
    computeFeatureReconstructionQuality(testBatchSize = 500, sparsity = 0.1, importanceDecay = 0.9) {
        const importanceVector = SuperpositionModel.computeImportanceVector(this.inputDim, importanceDecay);
        const featureErrors = new Array(this.inputDim).fill(0);
        const featureCounts = new Array(this.inputDim).fill(0);
        
        // Generate test batch and compute reconstruction errors
        for (let i = 0; i < testBatchSize; i++) {
            const x = Vector.sparse(this.inputDim, sparsity);
            
            // Normalize
            const norm = Vector.norm(x);
            if (norm > 0) {
                for (let j = 0; j < x.length; j++) {
                    x[j] /= norm;
                }
            }
            
            const { x_reconstructed } = this.forward(x);
            
            // Track error for each active feature
            for (let j = 0; j < this.inputDim; j++) {
                if (x[j] > 0) {  // Feature was active
                    const error = Math.abs(x[j] - x_reconstructed[j]);
                    featureErrors[j] += error;
                    featureCounts[j]++;
                }
            }
        }
        
        // Compute average reconstruction quality per feature
        const qualities = new Array(this.inputDim);
        for (let i = 0; i < this.inputDim; i++) {
            if (featureCounts[i] > 0) {
                const avgError = featureErrors[i] / featureCounts[i];
                qualities[i] = Math.max(0, 1 - avgError);  // Convert error to quality
            } else {
                qualities[i] = 0;  // No data for this feature
            }
        }
        
        return {
            qualities,
            importanceVector,
            featureCounts
        };
    }
}