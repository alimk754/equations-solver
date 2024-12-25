from sympy import symbols, lambdify
import time
def round_significant_figures(x: float, sig_figs: int = -1) -> float:
    if sig_figs == -1 or x == 0:
        return x
    return float(f'{x:.{sig_figs}g}')

class SecantMethod:
    def __init__(
        self,
        equation,
        x0: float,
        x1: float,
        max_error: float = 1e-5,
        significant_figures: int = -1,
        max_iterations: int = 50
    ):
        self.x0 = x0
        self.x1 = x1
        self.max_error = max_error
        self.significant_figures = significant_figures
        self.max_iterations = max_iterations
        
        x_symbols = symbols('x')
        self.func = lambdify(x_symbols, equation)
    
    def __next__(self, x0: float, x1: float) -> float:
        f0 = round_significant_figures(self.func(x0), self.significant_figures)
        f1 = round_significant_figures(self.func(x1), self.significant_figures)
        
        if f0 == f1:
            return x1
            
        numerator = (x1 - x0) * f0
        denominator = f1 - f0
        
        if self.significant_figures != -1:
            numerator = round_significant_figures(numerator, self.significant_figures)
            denominator = round_significant_figures(denominator, self.significant_figures)
            
        return round_significant_figures(
            x0 - numerator / denominator,
            self.significant_figures
        )
    
    def get_root(self) -> float:
        x0, x1 = self.x0, self.x1
        status = 400
        error = None
        current_time = time.time()
        for iteration in range(1, self.max_iterations + 1):
                x_new = self.__next__(x0, x1)
                error = abs((x_new - x1) / x_new) if x_new != 0 else float('inf')
                
                x0, x1 = x1, x_new

                if error <= self.max_error:
                    status = 200
                    break
                
        return {
            'status': 'Method did not converge within the maximum number of iterations.' 
                    if status == 400 else 'Method converged successfully.',
            'root': x1,
            'error': error,
            'iteration': iteration,
            'excution_time': time.time() - current_time
        }


secant_solver = SecantMethod(
        equation="x**5 - x - 1",
        x0=0,
        x1=1.15,
        max_iterations=100,
    significant_figures=12
)

result = secant_solver.get_root()
print(result)