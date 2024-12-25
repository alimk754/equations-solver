import sympy as sp

class RegulaFalsePosition:
    def __init__(self, precision=5, tol=1e-4, max_iter=50):
        self.precision = precision
        self.tol = tol
        self.max_iter = max_iter
        self.x = sp.symbols('x')
    
    def relative_error(self, x1, x2):
        if x1 == 0 or x2 == 0:
            return 17
        return abs((x2-x1)/x2)
    
    def check_validity(self, x):
        return x == sp.zoo or not x.is_real or not x.is_finite
    
    def formula_calc(self, equ, a, b):
        fa = equ.subs(self.x, a)
        fb = equ.subs(self.x, b)
        return (a*fb - b*fa)/(fb - fa)
    
    def false_position(self, equ, a, b):
        iter_count = self.max_iter
        a = sp.N(a, self.precision)
        b = sp.N(b, self.precision)
        
        if equ.subs(self.x, a) == 0:
            yield {
                "iteration": 0,
                "oldRoot": "none",
                "newRoot": a,
                "relativeError": "no relative error"
            }
            return
            
        if equ.subs(self.x, b) == 0:
            yield {
                "iteration": 0,
                "oldRoot": "none",
                "newRoot": b,
                "relativeError": "no relative error"
            }
            return
            
        if equ.subs(self.x, a) > 0:
            tmp = sp.N(b, self.precision)
            b = sp.N(a, self.precision)
            a = tmp
            
        prev = 0
        
        if self.check_validity(equ.subs(self.x, a)) or self.check_validity(equ.subs(self.x, b)):
            raise Exception("the function is not continuous at the given range")
            
        while iter_count >= 0:
            
            mid = self.formula_calc(equ, a, b)
            
            if self.check_validity(equ.subs(self.x, mid)):
                raise Exception("the function is not continuous at the given range")
                
            if equ.subs(self.x, a) * equ.subs(self.x, b) > 0:
                raise Exception("this equation cannot be solved by the False Position method")
                
            if equ.subs(self.x, mid) == 0:
                relative = sp.N(self.relative_error(mid, prev), self.precision)
                if relative == 17:
                    relative = "no relative error"
                yield {
                    "iteration": self.max_iter - iter_count,
                    "oldRoot": prev,
                    "newRoot": mid,
                    "relativeError": relative
                }
                break
                
            if equ.subs(self.x, mid) > 0:
                b = mid
            else:
                a = mid
                
            relative = self.relative_error(mid, prev)
            if relative == 17:
                relative = "no relative error"
                
            yield {
                "iteration": self.max_iter - iter_count,
                "oldRoot": prev,
                "newRoot": mid,
                "relativeError": relative
            }
            
            if self.relative_error(prev, mid) < self.tol:
                break
                
            prev = mid
            
        if iter_count == 0:
            raise Exception("method didn't converge")
        iter_count -= 1
    
    def final_result(self, equ, a, b):
        generator = self.false_position(equ, a, b)
        final_result = 0
        for step in generator:
            final_result = step["newRoot"]
        return final_result

def main():
    solver = FalsePositionMethod()
    x = sp.symbols('x')
    equation = sp.N(sp.sympify("x**2+6*x+4"), solver.precision)
    g = equation - x
    
    generator = solver.false_position(g, -2, 5)
    for i in generator:
        print(i["iteration"])
        print(i["oldRoot"])
        print(i["newRoot"])
        print(i["relativeError"])
        print("--------------------------------")
    print(solver.final_result(g,-2,5))    

if __name__ == "__main__":
    main()