class Matrix {
    static zeros(rows, cols) {
        return Array(rows).fill(null).map(() => Array(cols).fill(0));
    }
    
    static random(rows, cols, scale = 1.0) {
        return Array(rows).fill(null).map(() => 
            Array(cols).fill(null).map(() => (Math.random() - 0.5) * 2 * scale)
        );
    }
    
    static randomNormal(rows, cols, mean = 0, std = 1) {
        return Array(rows).fill(null).map(() => 
            Array(cols).fill(null).map(() => {
                let u = 0, v = 0;
                while (u === 0) u = Math.random();
                while (v === 0) v = Math.random();
                return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            })
        );
    }
    
    static multiply(A, B) {
        const rowsA = A.length;
        const colsA = A[0].length;
        const colsB = B[0].length;
        const result = Matrix.zeros(rowsA, colsB);
        
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                for (let k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    static multiplyVector(matrix, vector) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = new Array(rows).fill(0);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
    
    static transpose(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Matrix.zeros(cols, rows);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    static add(A, B) {
        const rows = A.length;
        const cols = A[0].length;
        const result = Matrix.zeros(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        return result;
    }
    
    static subtract(A, B) {
        const rows = A.length;
        const cols = A[0].length;
        const result = Matrix.zeros(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        return result;
    }
    
    static scale(matrix, scalar) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Matrix.zeros(rows, cols);
        
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[i][j] = matrix[i][j] * scalar;
            }
        }
        return result;
    }
    
    static normalize(matrix) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Matrix.zeros(rows, cols);
        
        for (let j = 0; j < cols; j++) {
            let norm = 0;
            for (let i = 0; i < rows; i++) {
                norm += matrix[i][j] * matrix[i][j];
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (let i = 0; i < rows; i++) {
                    result[i][j] = matrix[i][j] / norm;
                }
            }
        }
        return result;
    }
}

class Vector {
    static zeros(size) {
        return new Array(size).fill(0);
    }
    
    static random(size, scale = 1.0) {
        return Array(size).fill(null).map(() => (Math.random() - 0.5) * 2 * scale);
    }
    
    static sparse(size, sparsity) {
        const vector = Vector.zeros(size);
        const numActive = Math.max(1, Math.floor(size * (1 - sparsity)));
        const indices = [];
        
        for (let i = 0; i < size; i++) indices.push(i);
        
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        for (let i = 0; i < numActive; i++) {
            vector[indices[i]] = Math.random();
        }
        
        return vector;
    }
    
    static add(a, b) {
        return a.map((val, i) => val + b[i]);
    }
    
    static subtract(a, b) {
        return a.map((val, i) => val - b[i]);
    }
    
    static scale(vector, scalar) {
        return vector.map(val => val * scalar);
    }
    
    static dot(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }
    
    static norm(vector) {
        return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    }
    
    static mse(a, b) {
        const diff = Vector.subtract(a, b);
        return Vector.dot(diff, diff) / a.length;
    }
    
    static relu(vector) {
        return vector.map(val => Math.max(0, val));
    }
    
    static l1Norm(vector) {
        return vector.reduce((sum, val) => sum + Math.abs(val), 0);
    }
}