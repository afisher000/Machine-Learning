# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:15:35 2024

@author: afisher
"""
import numpy as np

class Quadratic:
    def __init__(self):
        self.label = 'Quadratic'
        self.initial_value = (5,5)
    def evaluate(self, x, y):
        return 2*x**2 + y**2

class Exponential:
    def __init__(self):
        self.label = 'Exponential'
        self.initial_value = (5,4)
    def evaluate(self, x, y):
        return np.exp(x)-x + np.exp(y)-y

class Logarithmic:
    def __init__(self):
        self.label = 'Logarithmic'
        self.initial_value = (5,5)
    def evaluate(self, x, y):
        return np.log(x + 1) + np.log(y + 1)

class Rosenbrock:
    def __init__(self, a=1, b=100, scale=1):
        self.label = 'Rosenbrock'
        self.initial_value = (4, -4)
        self.a = a
        self.b = b
        self.scale = scale
    def evaluate(self, x, y):
        return self.scale * ( (self.a - x)**2 + self.b * (y - x**2)**2 )

class Himmelblau:
    def __init__(self, scale=1):
        self.label = 'Himmelblau'
        self.initial_value = (5,5)
        self.scale = scale
    def evaluate(self, x, y):
        return self.scale * ((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

class Rastrigin:
    def __init__(self, A=10, scale=1):
        self.label = 'Rastrigin'
        self.initial_value = (4.5,4)
        self.A = A
        self.scale = scale
    def evaluate(self, x, y):
        return self.scale * (self.A * 2 + (x**2 - self.A * np.cos(2 * np.pi * x)) + (y**2 - self.A * np.cos(2 * np.pi * y)))

class Ackley:
    def __init__(self, scale=1):
        self.label = 'Ackley'
        self.initial_value = (4.3,4)
        self.scale = scale
    def evaluate(self, x, y):
        return self.scale * (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e)
