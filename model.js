class SuperpositionModel {
    constructor(inputDim, hiddenDim, activation = 'linear') {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.activation = activation;
        
        // Initialize with Xavier/Glorot initialization
        const scale = Math.sqrt(2.0 / (inputDim + hiddenDim));
        this.W = Matrix.randomNormal(hiddenDim, inputDim, 0, scale);
        
        this.lossHistory = [];
        this.isTraining = false;
        this.totalSteps = 0;
    }
    
    forward(x) {
        let h = Matrix.multiplyVector(this.W, x);
        
        if (this.activation === 'relu') {
            h = Vector.relu(h);
        }
        
        const W_T = Matrix.transpose(this.W);
        const x_reconstructed = Matrix.multiplyVector(W_T, h);
        
        return { h, x_reconstructed };
    }
    
    computeLoss(x, x_reconstructed, h, sparsityWeight = 0.1) {
        const reconstructionLoss = Vector.mse(x, x_reconstructed);
        
        const sparsityLoss = Vector.l1Norm(h) / h.length;
        
        const totalLoss = reconstructionLoss + sparsityWeight * sparsityLoss;
        
        return {
            total: totalLoss,
            reconstruction: reconstructionLoss,
            sparsity: sparsityLoss
        };
    }
    
    backward(x, learningRate = 0.01, sparsityWeight = 0.1) {
        const { h, x_reconstructed } = this.forward(x);
        
        const loss = this.computeLoss(x, x_reconstructed, h, sparsityWeight);
        
        const reconstruction_grad = Vector.scale(Vector.subtract(x_reconstructed, x), 2 / x.length);
        
        const W_T = Matrix.transpose(this.W);
        let dL_dh = Matrix.multiplyVector(W_T, reconstruction_grad);
        
        const sparsity_grad = h.map(val => sparsityWeight * Math.sign(val) / h.length);
        dL_dh = Vector.add(dL_dh, sparsity_grad);
        
        if (this.activation === 'relu') {
            dL_dh = dL_dh.map((grad, i) => h[i] > 0 ? grad : 0);
        }
        
        const dL_dW = Matrix.zeros(this.hiddenDim, this.inputDim);
        for (let i = 0; i < this.hiddenDim; i++) {
            for (let j = 0; j < this.inputDim; j++) {
                dL_dW[i][j] = dL_dh[i] * x[j] + reconstruction_grad[j] * h[i];
            }
        }
        
        this.W = Matrix.subtract(this.W, Matrix.scale(dL_dW, learningRate));
        
        // Optional: Add weight decay for regularization
        // this.W = Matrix.scale(this.W, 1 - learningRate * 0.0001);
        
        return loss;
    }
    
    generateBatch(batchSize, sparsity, importance) {
        const batch = [];
        
        for (let i = 0; i < batchSize; i++) {
            const x = Vector.sparse(this.inputDim, sparsity);
            
            if (importance < 1.0) {
                for (let j = 0; j < x.length; j++) {
                    x[j] *= Math.pow(j / x.length, 1 - importance);
                }
            }
            
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
    
    trainStep(batch, learningRate, sparsityWeight) {
        let totalLoss = 0;
        
        for (const x of batch) {
            const loss = this.backward(x, learningRate, sparsityWeight);
            totalLoss += loss.total;
            this.totalSteps++;
        }
        
        return totalLoss / batch.length;
    }
    
    async train(params = {}) {
        const {
            epochs = 1000,
            batchSize = 32,
            initialLearningRate = 0.1,
            minLearningRate = 0.001,
            decayRate = 0.995,
            sparsity = 0.1,
            sparsityWeight = 0.1,
            importance = 1.0,
            convergenceThreshold = 1e-5,
            callback = null
        } = params;
        
        this.isTraining = true;
        this.lossHistory = [];
        this.totalSteps = 0;
        
        let previousLoss = Infinity;
        let convergenceCount = 0;
        let currentLearningRate = initialLearningRate;
        
        for (let epoch = 0; epoch < epochs && this.isTraining; epoch++) {
            // Exponential decay of learning rate
            currentLearningRate = Math.max(
                minLearningRate,
                initialLearningRate * Math.pow(decayRate, epoch)
            );
            
            const batch = this.generateBatch(batchSize, sparsity, importance);
            const loss = this.trainStep(batch, currentLearningRate, sparsityWeight);
            
            this.lossHistory.push(loss);
            
            // Check for convergence
            const relativeLossChange = Math.abs((loss - previousLoss) / (previousLoss + 1e-8));
            if (relativeLossChange < convergenceThreshold) {
                convergenceCount++;
                if (convergenceCount > 20) {
                    console.log(`Converged at epoch ${epoch}`);
                    break;
                }
            } else {
                convergenceCount = 0;
            }
            
            previousLoss = loss;
            
            if (callback && epoch % 10 === 0) {
                callback({
                    epoch,
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
            epochs: this.lossHistory.length,
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
}