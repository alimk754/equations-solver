import sympy as sp

class FixedPointIteration:
    def __init__(self, precision=5, tol=1e-4, max_iter=50):
        self.precision = precision
        self.tol = tol
        self.max_iter = max_iter
        self.x = sp.Symbol('x')
    
    def relative_error(self, x1, x2):
        if x1 == 0 or x2 == 0:
            return 17.1717
        return abs((x2-x1)/x2)
    
    def check_validity(self, x):
        return x == sp.zoo or not x.is_real or not x.is_finite
    
    def fixed_point_iteration(self, equ, start_point):
        iter_count = self.max_iter
        start_point = sp.N(start_point, self.precision)
        root = start_point
        diff = sp.diff(equ)
        
        while iter_count >= 0:
            if diff.subs(self.x, root) > 1:
                raise Exception("method doesn't converge")
                
            iter_count -= 1
            new_root = equ.subs(self.x, root)
            
            if self.check_validity(new_root):
                raise Exception("cannot be solved")
                
            relative = self.relative_error(root, new_root)
            
            if relative < self.tol:
                yield {
                    "iteration": self.max_iter - iter_count,
                    "x_i": root,
                    "x_i+1": new_root,
                    "RelativeError": relative
                }
                break
                
            if relative == 17.1717:
                relative = "no relative error"
                
            yield {
                "iteration": self.max_iter - iter_count,
                "x_i": root,
                "x_i+1": new_root,
                "RelativeError": relative
            }
            
            root = new_root
            
        if self.max_iter == 0:
            raise Exception("method didn't converge")
    
    def final_result(self, equ, start_point):
        generator = self.fixed_point_iteration(equ, start_point)
        final_result = 0
        for step in generator:
            final_result = step["x_i+1"]
        return final_result

def main():
    solver = FixedPointIteration(precision=6)
    equation = "exp(-x)-x"
    equation = sp.N(sp.sympify(equation), solver.precision)
    
    generator = solver.fixed_point_iteration(equation, 4)
    for i in generator:
        print(i)
    
    print(solver.final_result(equation, 4))

if __name__ == "__main__":
    main()