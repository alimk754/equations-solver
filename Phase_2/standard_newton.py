from sympy import diff, lambdify, symbols, sympify

class StandardNewton:
    def __init__(self, expression, precision=6):
        """
        Initialize the Newton-Raphson solver with the given expression.

        :param expression: The function as a string to solve for roots.
        :param precision: The numerical precision for calculations.
        """
        var = symbols('x')
        self.var = var
        self.precision = precision
        self.expression = sympify(expression)
        self.f = lambdify(var, self.expression, 'numpy')
        self.df = lambdify(var, diff(self.expression, var), 'numpy')

    def iter_steps(self, initial_guess, tolerance=1e-6, max_iter=100):
        """
        Generate the Newton-Raphson iteration steps.

        :param initial_guess: The starting point for the iterations.
        :param tolerance: The stopping criterion for function value and step size.
        :param max_iter: The maximum number of iterations allowed.
        :yield: A dictionary containing iteration details.
        """
        x = float(initial_guess)  # Ensure numeric input

        for i in range(max_iter):
            fx = self.f(x)  # Evaluate f(x)
            dfx = self.df(x)  # Evaluate f'(x)

            if abs(dfx) < 1e-12:
                raise RuntimeError(f"Derivative too small at iteration {i}: dfx = {dfx}")

            x_new = x - fx / dfx  # Newton-Raphson step

            yield {
                'iteration': i,
                'x_i': round(x, self.precision),
                'x_i+1': round(x_new, self.precision),
                'fx': round(fx, self.precision),
                'dfx': round(dfx, self.precision),
                'absolute error': None if i == 0 else round(abs(x_new - x), self.precision),
                'relative error': None if x_new == 0 else round(abs(x_new - x) / abs(x_new), self.precision)
            }

            if abs(fx) < tolerance or abs(x_new - x) < tolerance:
                break

            x = x_new

    def find_root(self, initial_guess, tolerance=1e-6, max_iter=100):
        """
        Find a root using the Newton-Raphson method.

        :param initial_guess: The starting point for the iterations.
        :param tolerance: The stopping criterion for function value and step size.
        :param max_iter: The maximum number of iterations allowed.
        :return: The approximated root value.
        """
        x = float(initial_guess)  # Ensure numeric input

        for i in range(max_iter):
            fx = self.f(x)  # Evaluate f(x)
            dfx = self.df(x)  # Evaluate f'(x)

            if abs(dfx) < 1e-12:
                raise RuntimeError(f"Derivative too small at iteration {i}: dfx = {dfx}")

            x_new = x - fx / dfx  # Newton-Raphson step

            if abs(fx) < tolerance or abs(x_new - x) < tolerance:
                return round(x_new, self.precision)

            x = x_new

        raise RuntimeError(f"Failed to converge after {max_iter} iterations.")