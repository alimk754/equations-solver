from PyQt6.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout
from sympy import N
class GeneratorWindow(QMainWindow):
    def __init__(self, generator,precision):
        super().__init__()
        self.generator = generator
        self.steps = []
        self.current_step = -1
        self.init_ui()
        self.precision=precision
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("font-family: monospace;")
        
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton('Previous')
        self.prev_button.clicked.connect(self.show_prev)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.show_next)
        
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        
        main_layout.addWidget(self.text_display)
        main_layout.addLayout(button_layout)
        
        self.setWindowTitle('Numerical Method Steps')
        self.resize(500, 400)
        
    def format_dict(self, d):
        lines = []
        for key, value in d.items():
            if hasattr(value, 'evalf'):
                value = N(value.evalf(),self.precision)
            lines.append(f"{key:15} : {value}")
        return '\n'.join(lines)
        
    def show_next(self):
        try:
            if self.current_step == len(self.steps) - 1:
                value = next(self.generator)
                self.steps.append(value)
            self.current_step += 1
            
            self.text_display.clear()
            value = self.steps[self.current_step]
            if isinstance(value, dict):
                formatted = self.format_dict(value)
                self.text_display.append(formatted)
            else:
                self.text_display.append(str(value))
                
            self.prev_button.setEnabled(self.current_step > 0)
        except StopIteration:
            self.next_button.setEnabled(False)
            self.text_display.append('Process completed')
            
    def show_prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.text_display.clear()
            value = self.steps[self.current_step]
            if isinstance(value, dict):
                formatted = self.format_dict(value)
                self.text_display.append(formatted)
            else:
                self.text_display.append(str(value))
            
            self.prev_button.setEnabled(self.current_step > 0)
            self.next_button.setEnabled(True)