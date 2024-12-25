import sys

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton,
                             QSpinBox, QDoubleSpinBox, QApplication)
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sympy import symbols, simplify

from Bisection import Bisection
from FixedPointIteration import FixedPointIteration
from RegulaFalseMethod import RegulaFalsePosition
from modified_1_newton import Modified1Newton
from modified_2_newton import Modified2Newton
from standard_newton import StandardNewton
from Secant import SecantMethod
from step import GeneratorWindow


class Phase2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Finder - Phase 2")
        self.setMinimumSize(800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Input Section
        input_group = QWidget()
        input_layout = QHBoxLayout(input_group)

        # Equation input
        equation_label = QLabel("Enter equation:")
        self.equation_input = QLineEdit()
        self.equation_input.setPlaceholderText("e.g., x^2 - 4*x + 4")
        input_layout.addWidget(equation_label)
        input_layout.addWidget(self.equation_input)

        # Method selection
        method_label = QLabel("Select Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Bisection",
            "False-Position",
            "Fixed Point",
            "Newton-Raphson",
            "Modified 1 Newton-Raphson",
            "Modified 2 Newton-Raphson",
            "Secant Method"
        ])
        input_layout.addWidget(method_label)
        input_layout.addWidget(self.method_combo)

        layout.addWidget(input_group)

        # Parameters Section
        params_group = QWidget()
        params_layout = QHBoxLayout(params_group)

        # Interval/Initial guess inputs
        self.interval_label = QLabel("Interval:")
        self.a_input = QDoubleSpinBox()
        self.a_input.setRange(-1000, 1000)
        self.b_input = QDoubleSpinBox()
        self.b_input.setRange(-1000, 1000)
        self.gx_label = QLabel("g(x):")
        self.gx = QLineEdit()
        self.gx.setPlaceholderText("e.g., x^2 - 4*x + 4")
        self.gx_label.hide()
        self.gx.hide()

        params_layout.addWidget(self.interval_label)
        params_layout.addWidget(self.a_input)
        params_layout.addWidget(self.b_input)
        params_layout.addWidget(self.gx_label)
        params_layout.addWidget(self.gx)

        # Precision settings
        precision_label = QLabel("Precision:")
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(1, 15)
        self.precision_spin.setValue(6)

        eps_label = QLabel("Epsilon:")
        self.eps_input = QDoubleSpinBox()
        self.eps_input.setDecimals(6)
        self.eps_input.setValue(0.00001)
        self.eps_input.setRange(1e-10, 1)

        max_iter_label = QLabel("Max Iterations:")
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(50)

        params_layout.addWidget(precision_label)
        params_layout.addWidget(self.precision_spin)
        params_layout.addWidget(eps_label)
        params_layout.addWidget(self.eps_input)
        params_layout.addWidget(max_iter_label)
        params_layout.addWidget(self.max_iter_spin)

        layout.addWidget(params_group)

        # Plot Area
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Results Section
        results_group = QWidget()
        results_layout = QVBoxLayout(results_group)

        self.result_label = QLabel("Results will appear here")
        results_layout.addWidget(self.result_label)

        # Buttons
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)

        self.plot_button = QPushButton("Plot Function")
        self.solve_button = QPushButton("Solve")
        self.step_button = QPushButton("Step Mode")  # Bonus feature

        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.solve_button)
        button_layout.addWidget(self.step_button)

        results_layout.addWidget(button_group)

        plot_input = QWidget()
        plot_input_layout = QHBoxLayout(plot_input)

        plot_start_label = QLabel("plot start")
        self.plot_start = QSpinBox()
        self.plot_start.setRange(-1000, 1000)
        self.plot_start.setValue(-10)

        plot_end_label = QLabel("plot end")
        self.plot_end = QSpinBox()
        self.plot_end.setRange(-1000, 1000)
        self.plot_end.setValue(10)

        plot_input_layout.addWidget(plot_start_label)
        plot_input_layout.addWidget(self.plot_start)
        plot_input_layout.addWidget(plot_end_label)
        plot_input_layout.addWidget(self.plot_end)

        results_layout.addWidget(plot_input)
        layout.addWidget(results_group)

        # Connect signals
        self.plot_button.clicked.connect(self.plot_function)
        self.solve_button.clicked.connect(self.solve_equation)
        self.step_button.clicked.connect(self.step_mode)
        self.method_combo.currentTextChanged.connect(self.update_input_fields)

    def plot_function(self):
        try:
            # Clear the previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Get the equation and create x values
            equation = self.equation_input.text().strip().lower()

            # Replace common mathematical notations
            equation = equation.replace('^', '**')  # Convert ^ to **
            equation = equation.replace('e**x', 'exp(x)')  # Handle e^x notation

            # Define x range
            st = self.plot_start.value()
            end = self.plot_end.value()
            if st > end:
                temp = st
                st = end
                end = temp
            x = np.linspace(st, end, (end - st) * 100)

            # Create safe evaluation environment
            safe_dict = {
                'x': x,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'log': np.log,
                'sqrt': np.sqrt,
                'pi': np.pi,
                'e': np.e,
                'abs': np.abs
            }

            # Validate equation
            valid_chars = set('x0123456789.+-*/() ')
            valid_funcs = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt'}

            # Check for valid characters and functions
            cleaned_eq = equation
            for func in valid_funcs:
                cleaned_eq = cleaned_eq.replace(func, '')
            if not all(c in valid_chars for c in cleaned_eq):
                raise ValueError("Invalid characters in equation")

            # Calculate y values
            # Using eval in a controlled environment with only math functions
            y = eval(equation, {"__builtins__": {}}, safe_dict)

            # Find reasonable y-axis limits
            finite_mask = np.isfinite(y)
            if np.any(finite_mask):
                y_finite = y[finite_mask]
                if len(y_finite) > 0:
                    y_min = np.min(y_finite)
                    y_max = np.max(y_finite)
                    y_range = y_max - y_min
                    ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

            # Plot the function
            ax.plot(x, y, label='f(x)')

            # For Fixed Point method, add y=x line
            if self.method_combo.currentText() == "Fixed Point":
                ax.plot(x, x, '--', label='y=x')

            # Add grid and labels
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Plot of {equation}')
            ax.legend()

            # Add x and y axis lines
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

            # Update the canvas
            self.canvas.draw()

            # Clear any previous error messages
            self.result_label.setText("Function plotted successfully")

        except Exception as e:
            self.result_label.setText(f"Error plotting function: {str(e)}\nPlease check your equation syntax.")

    def get_roots_in_view(self, x, y):
        """Helper function to find approximate roots visible in the plot"""
        roots = []
        for i in range(len(y) - 1):
            if y[i] * y[i + 1] <= 0:
                # Linear interpolation to find more precise root
                root = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                roots.append(round(root, 4))
        return roots

    def solve_equation(self):
        # TODO: Implement the selected numerical method
        global equation
        method = self.method_combo.currentText()

        self.result_label.setText(f"Solving using {method}...")

        input_equation = self.equation_input.text()
        gx = self.gx.text()
        formatted_equation = input_equation.replace("^", "**")
        gxf = gx.replace("^", "**")

        try:
            equation = simplify(formatted_equation)
        except Exception as e:
            self.result_label.setText(f"Error solving function: {str(e)}\nPlease check your equation syntax.")

        try:
            if method == "Bisection":
                solver = Bisection(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
                ans = solver.final_result(equation, self.a_input.value(), self.b_input.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "False-Position":
                solver = RegulaFalsePosition(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
                ans = solver.final_result(equation, self.a_input.value(), self.b_input.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "Fixed Point":
                gxsim = simplify(gxf)
                solver = FixedPointIteration(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
                ans = solver.final_result(gxsim, self.a_input.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "Newton-Raphson":
                solver = StandardNewton(equation, self.precision_spin.value())
                ans = solver.find_root(self.a_input.value(), self.eps_input.value(), self.max_iter_spin.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "Modified 1 Newton-Raphson":
                solver = Modified1Newton(equation, self.precision_spin.value())
                ans = solver.find_root(self.a_input.value(), self.b_input.value(),
                                          self.eps_input.value(), self.max_iter_spin.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "Modified 2 Newton-Raphson":
                solver = Modified2Newton(equation, self.precision_spin.value())
                ans = solver.find_root(self.a_input.value(), self.eps_input.value(), self.max_iter_spin.value())
                self.result_label.setText(f"found root = {ans}")

            elif method == "Secant Method":
                solver = SecantMethod(equation, self.a_input.value(),self.b_input.value(),self.eps_input.value(),self.precision_spin.value(),self.max_iter_spin.value())
                res=solver.get_root() 
                self.result_label.setText(f"found root = {res['root']}")

        except Exception as e:
            self.result_label.setText(f"Error solving function: {str(e)}")


    def step_mode(self):
        method = self.method_combo.currentText()
    
        if method == "Bisection":
            solver = Bisection(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
            generator = solver.bisection(equation, self.a_input.value(), self.b_input.value())
    
        elif method == "False-Position":
            solver = RegulaFalsePosition(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
            generator = solver.false_position(equation, self.a_input.value(), self.b_input.value())
    
        elif method == "Fixed Point":
            gx = self.gx.text().replace("^", "**")
            input_equation = self.equation_input.text().replace("^", "**")
            gxsim = simplify(gx)
            solver = FixedPointIteration(self.precision_spin.value(), self.eps_input.value(), self.max_iter_spin.value())
            generator = solver.fixed_point_iteration(gxsim, self.a_input.value())
        elif method == "Newton-Raphson":
            solver = StandardNewton(equation, self.precision_spin.value())
            generator = solver.iter_steps(self.a_input.value(), self.eps_input.value(), self.max_iter_spin.value())
                
        elif method == "Modified 1 Newton-Raphson":
            solver = Modified1Newton(equation, self.precision_spin.value())
            generator = solver.iter_steps(self.a_input.value(), self.b_input.value(),
                                          self.eps_input.value(), self.max_iter_spin.value())
                

        elif method == "Modified 2 Newton-Raphson":
            solver = Modified2Newton(equation, self.precision_spin.value())
            generator = solver.iter_steps(self.a_input.value(), self.eps_input.value(), self.max_iter_spin.value())
    
    
        self.step_window = GeneratorWindow(generator,self.precision_spin.value())
        self.step_window.show()  

    def update_input_fields(self, method):
        # Update input fields based on selected method
        if method in ["Bisection", "False-Position"]:
            self.interval_label.setText("Interval [a, b]:")
            self.a_input.show()
            self.b_input.show()
            self.gx.hide()
            self.gx_label.hide()
        elif method in ["Newton-Raphson","Modified 2 Newton-Raphson"]:
            self.interval_label.setText("Initial guess:")
            self.a_input.show()
            self.b_input.hide()
            self.gx.hide()
            self.gx_label.hide()
        elif method  == "Fixed Point":
            self.interval_label.setText("initial guess:")
            self.a_input.show()
            self.b_input.hide()
            self.gx.show()
            self.gx_label.show()
        else:  # Secant Method and modified newton 1
            self.interval_label.setText("Initial guesses:")
            self.a_input.show()
            self.b_input.show()
            self.gx.hide()
            self.gx_label.hide()

def main():
    app = QApplication(sys.argv)
    window = Phase2Window()
    window.show()
    sys.exit(app.exec())

main()