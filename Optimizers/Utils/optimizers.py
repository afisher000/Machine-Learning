# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:17:50 2024

@author: afisher
"""
import numpy as np

class Adam():
    def __init__(self, label='Adam', learning_rate=0.01, beta1=0.9, beta2=0.9, epsilon=1e-8):
        self.label = label
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_x = 0
        self.v_x = 0
        self.m_y = 0
        self.v_y = 0
        self.t = 0
    
    def simulate(self, func, iterations=10):
        x, y = func.initial_value
        history = [(x, y)]
        
        for _ in range(iterations):
            self.t += 1
            grad_x = (func.evaluate(x + self.epsilon, y) - func.evaluate(x, y)) / self.epsilon
            grad_y = (func.evaluate(x, y + self.epsilon) - func.evaluate(x, y)) / self.epsilon
            
            # Update biased first moment estimate
            self.m_x = self.beta1 * self.m_x + (1 - self.beta1) * grad_x
            self.m_y = self.beta1 * self.m_y + (1 - self.beta1) * grad_y
            
            # Update biased second raw moment estimate
            self.v_x = self.beta2 * self.v_x + (1 - self.beta2) * grad_x**2
            self.v_y = self.beta2 * self.v_y + (1 - self.beta2) * grad_y**2
            
            # Correct bias in first moment estimate
            m_x_hat = self.m_x / (1 - self.beta1**self.t)
            m_y_hat = self.m_y / (1 - self.beta1**self.t)
            
            # Correct bias in second moment estimate
            v_x_hat = self.v_x / (1 - self.beta2**self.t)
            v_y_hat = self.v_y / (1 - self.beta2**self.t)
            
            # Update parameters
            x -= (self.learning_rate / (np.sqrt(v_x_hat) + self.epsilon)) * m_x_hat
            y -= (self.learning_rate / (np.sqrt(v_y_hat) + self.epsilon)) * m_y_hat
            
            history.append((x, y))
        
        return history


class Adagrad():
    def __init__(self, label='Adagrad', learning_rate=0.01):
        self.label = label
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.cumulative_gradients_x = 0
        self.cumulative_gradients_y = 0
    
    def simulate(self, func, iterations=10):
        x, y = func.initial_value
        history = [(x, y)]
        
        for _ in range(iterations):
            grad_x = (func.evaluate(x + self.epsilon, y) - func.evaluate(x, y)) / self.epsilon
            grad_y = (func.evaluate(x, y + self.epsilon) - func.evaluate(x, y)) / self.epsilon
            
            # Accumulate squared gradients
            self.cumulative_gradients_x += grad_x ** 2
            self.cumulative_gradients_y += grad_y ** 2
            
            # Update parameters
            x -= (self.learning_rate / (np.sqrt(self.cumulative_gradients_x) + self.epsilon)) * grad_x
            y -= (self.learning_rate / (np.sqrt(self.cumulative_gradients_y) + self.epsilon)) * grad_y
            
            history.append((x, y))
        
        return history


class GradientDescent():
    def __init__(self, label='Gradient Descent', learning_rate=0.01):
        self.label = label
        self.learning_rate = learning_rate
        return
        
    def simulate(self, func, iterations=10):
        x, y = func.initial_value
        history = [(x, y)]
        eps = 1e-5
        
        for _ in range(iterations):
            grad_x = (func.evaluate(x + eps, y) - func.evaluate(x, y)) / eps
            grad_y = (func.evaluate(x, y + eps) - func.evaluate(x, y)) / eps
            x -= self.learning_rate * grad_x
            y -= self.learning_rate * grad_y
            history.append((x, y))
        return history


class Momentum():
    def __init__(self, label='Momentum', learning_rate=0.01, momentum=0.9):
        self.label = label
        self.learning_rate = learning_rate
        self.momentum = momentum
        return
    
    def simulate(self, func, iterations=10):
        x, y = func.initial_value
        history = [(x, y)]
        eps = 1e-5
            
        v_x, v_y = 0, 0  # Initial velocity
        for _ in range(iterations):
            grad_x = (func.evaluate(x + eps, y) - func.evaluate(x, y)) / eps
            grad_y = (func.evaluate(x, y + eps) - func.evaluate(x, y)) / eps
            
            # Update velocity
            v_x = self.momentum * v_x - self.learning_rate * grad_x
            v_y = self.momentum * v_y - self.learning_rate * grad_y
            
            # Update parameters
            x += v_x
            y += v_y
            
            history.append((x, y))
        return history


class NAG():
    def __init__(self, label='NAG', learning_rate=0.01, momentum=0.9):
        self.label = label
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def simulate(self, func, iterations=10):
        x, y = func.initial_value
        history = [(x, y)]
        eps = 1e-5
        v_x, v_y = 0, 0  # Initial velocity
        
        for _ in range(iterations):
            # Compute gradient at future position
            grad_x_future = (func.evaluate(x - self.momentum * v_x + eps, y - self.momentum * v_y) - func.evaluate(x - self.momentum * v_x, y - self.momentum * v_y)) / eps
            grad_y_future = (func.evaluate(x - self.momentum * v_x, y - self.momentum * v_y + eps) - func.evaluate(x - self.momentum * v_x, y - self.momentum * v_y)) / eps
            
            # Update velocity
            v_x = self.momentum * v_x - self.learning_rate * grad_x_future
            v_y = self.momentum * v_y - self.learning_rate * grad_y_future
            
            # Update parameters
            x += v_x
            y += v_y
            
            history.append((x, y))
        
        return history
