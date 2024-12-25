import numpy as np
import Forward_elimination

class gauss:
    def __init__(self, n, b):
        self.array = np.zeros((len(n), len(n)+1))
        for i in range(len(n)):
            for j in range(len(n)):
                self.array[i][j] = n[i][j]
        for i in range(len(n)):
            self.array[i][len(n)] = b[i]
        
        self.answer = np.zeros(len(self.array), dtype=float)
        self.scalers = self.scaling()
        self.eliminator = Forward_elimination.forward_eliminator(self.array, self.scalers)
        self.constant_vector = b.copy()
        self.coefficient_matrix=n.copy()
        self.solution_type = self.detect_system_type()
       
    def detect_system_type(self):
        A = self.coefficient_matrix
        Aug = np.column_stack((A, self.constant_vector))
        rank_A = np.linalg.matrix_rank(A)
        rank_Aug = np.linalg.matrix_rank(Aug)
        n = A.shape[1]
        
        if rank_A == rank_Aug:
            if rank_A == n:
                self.solution_type = 'unique'
                return 'unique'
            else:
                self.solution_type = 'infinite'
                return 'infinite'

        else:
            self.solution_type = 'no solution'
            return 'no solution'    

    def gauss_elimination_generator(self):
        if(self.solution_type=='no solution'):
            yield{
                "error":'there is no solution'
            }
            raise Exception("there is no solution")
      
        if(self.solution_type=='infinite'):
            yield{
                "error":'there is an infinite number of solution'
                
            }    
            raise Exception("there is an infinite number of solution")
      
       
        yield {
            'step': 'Initial Augmented Matrix',
            'matrix': self.array.copy(),
            'description': 'Starting augmented matrix before elimination'
        }

        # Scaling step
        yield {
            'step': 'Scaling',
            'scalers': self.scalers,
            'description': 'Computed row scaling factors'
        }

        # Elimination process
        while True:
            # Perform next elimination step
            elimination_result = self.eliminator.Forward_Elimination_nextStep()
            
            # Check if elimination is complete or failed
            if elimination_result == -1:
                yield {
                    'step': 'Elimination Failed',
                    'description': 'Unable to continue elimination process'
                }
                break
            
            # Unpack elimination result
            flag, updated_matrix = elimination_result
            
            # Yield current matrix state
            yield {
                'step': 'Elimination Step',
                'matrix': updated_matrix,
                'flag': flag,
                'description': 'Matrix after elimination step'
            }
            
            # If elimination is complete, perform backward substitution
            if flag:
                # Backward substitution
                self.array = updated_matrix
                final_answer = self.Backward_Substitution()
                
                yield {
                    'step': 'Backward Substitution',
                    'solution': final_answer,
                    'description': 'Final solution vector'
                }
                break

    def scaling(self):
        """Compute scaling factors for rows"""
        arr = self.array
        scalers = [0] * len(arr)
        
        for i in range(len(arr)):
            max_val = abs(arr[i][0])
            for j in range(1, len(arr)):
                if abs(arr[i][j]) > max_val:
                    max_val = abs(arr[i][j])
            
            scalers[i] = max_val
        
        return scalers

    def Backward_Substitution(self):
        """Perform backward substitution"""
        arr = self.array
        col = len(arr[0])
        rows = len(arr)
        
        self.answer[rows-1] = arr[rows-1][col-1] / arr[rows-1][rows-1]
        
        for i in range(rows-2, -1, -1):
            sum_val = 0
            for j in range(i+1, rows):
                sum_val += arr[i][j] * self.answer[j]
            
            self.answer[i] = (arr[i][col-1] - sum_val) / arr[i][i]
        
        return self.answer

    def Gauss(self):
        """Perform complete Gauss elimination"""
        if self.eliminator.Forward_Elimination() != -1:
            self.answer = self.Backward_Substitution()
            self.factors = self.eliminator.factors
        else:
            return -1
        
        return np.array(self.answer)

# Example usage
def main():
    # Create Gauss elimination object
    m1 = gauss(
        np.array([[25,5,1], [25,5,1], [144,12,1]], dtype=float), 
        np.array([1,1,3], dtype=float)
    )
    
    # Demonstrate generator usage
    print("Gauss Elimination Steps:")
    for step in m1.gauss_elimination_generator():
        print(step)

if __name__ == "__main__":
    main()

  










    

