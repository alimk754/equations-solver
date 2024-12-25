import numpy as np

class IterativeSolver:
    def __init__(self, A, b, x0=None):
        A = np.array(A, dtype=float)
        
        # Check for zero diagonal elements
        for i in range(A.shape[0]):
            if A[i, i] == 0:
                raise RuntimeError(f"Zero diagonal element found at index {i}.")
        
        self.A = A
        self.b = np.array(b, dtype=float)
        
        # Initial solution
        self.x = np.zeros_like(b, dtype=float) if x0 is None else np.array(x0, dtype=float)

    def solve_by_iterations(self, num_iterations):
        for _ in range(num_iterations):
            self.nextStep()
        return self.x

    def solve_by_error(self, error_threshold,max_iterations=10000):

        for _ in range(max_iterations):
            prev_x = self.x.copy()
            self.nextStep()
            
            error = np.linalg.norm(self.x - prev_x) / (np.linalg.norm(self.x))

            if (error < error_threshold):
                return self.x
             
        raise ValueError("iterative method did not converge within the maximum number of iterations.")

class JacobiSolver(IterativeSolver):
    def nextStep(self):
        n = len(self.b)
        new_x = np.zeros_like(self.x)
        
        for i in range(n):
            summation = sum(
                self.A[i][j] * self.x[j] 
                for j in range(n) if j != i
            )
            new_x[i] = (self.b[i] - summation) / self.A[i][i]
        
        self.x = new_x

class GaussSeidelSolver(IterativeSolver):
    def nextStep(self):
        n = len(self.b)
        
        for i in range(n):
            summation = sum(
                self.A[i][j] * self.x[j] 
                for j in range(n) if j != i
            )
            self.x[i] = (self.b[i] - summation) / self.A[i][i]