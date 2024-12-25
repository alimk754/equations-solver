from sympy import diff, lambdify, N, symbols, sympify


class Modified2Newton:

    def __init__(self, expression, var, precision=6):
        expression = N(expression, precision)
        var = symbols('x')
        self.expression = sympify(expression)
        self.precision = precision
        self.f = lambdify(var, expression)
        self.df = lambdify(var, diff(expression, var))
        self.d2f = lambdify(var, diff(expression, var, 2))

    def iter_steps(self, initial_guess, tolerance=1e-6, max_iter=100):
        initial_guess = N(initial_guess, self.precision)
        x = N(initial_guess, self.precision)

        for i in range(max_iter):
            fx = N(self.f(x), self.precision)
            dfx = N(self.df(x), self.precision)
            d2fx = N(self.d2f(x), self.precision)

            denom = N(dfx ** 2 - fx * d2fx, self.precision)
            x_new = N(x - (dfx * fx) / denom, self.precision)

            yield {
                'iteration': i,
                'x': x,
                'x_i+1': x_new,
                'fx': fx,
                'dfx': dfx,
                'd2fx': d2fx,
                'absolute error': None if i == 0 else abs(x_new - x),
                'relative error': None if x_new == 0 else abs(x_new - x) / x_new
            }

            if abs(fx) < tolerance:
                break
            if abs(x_new - x) < tolerance:
                break

            x = x_new

    def find_root(self, initial_guess, tolerance=1e-6, max_iter=100):
        initial_guess = N(initial_guess, self.precision)
        x = N(initial_guess, self.precision)

        for i in range(max_iter):
            fx = N(self.f(x), self.precision)
            dfx = N(self.df(x), self.precision)
            d2fx = N(self.d2f(x), self.precision)

            denom = N(dfx ** 2 - fx * d2fx, self.precision)
            x_new = N(x - (dfx * fx) / denom, self.precision)

            if abs(fx) < tolerance:
                return x
            if abs(x_new - x) < tolerance:
                return x_new

            x = x_new

        raise RuntimeError(f"Failed to converge after {max_iter} iterations")
