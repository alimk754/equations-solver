from sympy import diff, lambdify, symbols, sympify


class Modified2Newton:
    def __init__(self, expression, precision=6):
        """
        Initialize the Modified Newton-Raphson solver with second derivatives.

        :param expression: The function as a string to solve for roots.
        :param precision: The numerical precision for calculations.
        """
        var = symbols('x')
        self.var = var
        self.expression = sympify(expression)
        self.precision = precision
        
        # Lambdify the function, first derivative, and second derivative
        self.f = lambdify(var, self.expression, 'numpy')
        self.df = lambdify(var, diff(self.expression, var), 'numpy')
        self.d2f = lambdify(var, diff(self.expression, var, 2), 'numpy')

    def iter_steps(self, initial_guess, tolerance=1e-6, max_iter=100):
        """
        Generate the iteration steps for the Modified Newton-Raphson method.

        :param initial_guess: The starting point for the iterations.
        :param tolerance: The stopping criterion for function value and step size.
        :param max_iter: The maximum number of iterations allowed.
        :yield: A dictionary containing iteration details.
        """
        x = float(initial_guess)  # Ensure numeric input

        for i in range(max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            d2fx = self.d2f(x)

            # Compute the denominator for the update
            denom = dfx ** 2 - fx * d2fx
            if abs(denom) < 1e-12:
                raise RuntimeError(f"Denominator too small at iteration {i}: denom = {denom}")

            # Update using Modified Newton-Raphson formula
            x_new = x - (dfx * fx) / denom

            yield {
                'iteration': i,
                'x_i': round(x, self.precision),
                'x_i+1': round(x_new, self.precision),
                'fx': round(fx, self.precision),
                'dfx': round(dfx, self.precision),
                'd2fx': round(d2fx, self.precision),
                'absolute error': None if i == 0 else round(abs(x_new - x), self.precision),
                'relative error': None if x_new == 0 else round(abs(x_new - x) / abs(x_new), self.precision),
            }

            if abs(fx) < tolerance or abs(x_new - x) < tolerance:
                break

            x = x_new

    def find_root(self, initial_guess, tolerance=1e-6, max_iter=100):
        """
        Find a root using the Modified Newton-Raphson method.

        :param initial_guess: The starting point for the iterations.
        :param tolerance: The stopping criterion for function value and step size.
        :param max_iter: The maximum number of iterations allowed.
        :return: The approximated root value.
        """
        x = float(initial_guess)  # Ensure numeric input

        for i in range(max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            d2fx = self.d2f(x)

            # Compute the denominator for the update
            denom = dfx ** 2 - fx * d2fx
            if abs(denom) < 1e-12:
                raise RuntimeError(f"Denominator too small at iteration {i}: denom = {denom}")

            # Update using Modified Newton-Raphson formula
            x_new = x - (dfx * fx) / denom

            if abs(fx) < tolerance or abs(x_new - x) < tolerance:
                return round(x_new, self.precision)

            x = x_new

        raise RuntimeError(f"Failed to converge after {max_iter} iterations.")