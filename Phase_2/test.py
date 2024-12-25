from sympy import symbols, sin, cos, exp

from modified_1_newton import Modified1Newton
from modified_2_newton import Modified2Newton
from standard_newton import StandardNewton


def test():
    x = symbols('x')
    expr = x ** 3 - 2 * x ** 2 - 4 * x + 8
    newton = StandardNewton(expr, x)
    result = list(newton.iter_steps(1.2))
    print(result)

def test_standard_newton():
    x = symbols('x')
    expr = x ** (1/3);
    newton = StandardNewton(expr, x)
    result = newton.find_root(1)
    print(result)


def test_iter_steps_convergence():
    x = symbols('x')
    newton = StandardNewton(x ** 3 - 2 * x ** 2 - 4 * x + 8, x)
    steps = list(newton.iter_steps(1.2))

    # Check if steps are generating correctly
    assert len(steps) > 0
    assert all(isinstance(step, dict) for step in steps)
    # Check if final value converges to expected root
    final_x = steps[-1]['x_i+1']
    assert abs(final_x - 2.0) < 1e-6
