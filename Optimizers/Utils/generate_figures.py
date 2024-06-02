# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:01:05 2024

@author: afisher
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import Utils.cost_functions as cf
import Utils.optimizers as opt
plt.close('all')

def create_pic(func, optimizers, x_range=(-5, 5), y_range=(-5, 5), frames=50, filename=None):
    
    # Create figure
    plt.figure(figsize=(6, 4))
    
    # Plot cost function
    x = np.linspace(*x_range, 400)
    y = np.linspace(*y_range, 400)
    X, Y = np.meshgrid(x, y)
    Z = func.evaluate(X, Y)
    
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    
    # Plot optimizers
    for optimizer in optimizers:
        history = optimizer.simulate(func, iterations = frames)
        hx, hy = zip(*history)
        plt.plot(hx, hy, marker='o', markersize=5, label=optimizer.label)
        
    
    # Add properties
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.title(func.label)
    plt.legend(loc='upper left')
    # plt.colorbar()
    
    if filename is None:
        filename = func.label + ' ' + optimizer.label + '.jpg'
    plt.savefig(os.path.join('pics', filename))
    # plt.close()



def create_gif(func, optimizers, x_range=(-5, 5), y_range=(-5, 5), frames=50, filename=None):
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot cost function
    x = np.linspace(*x_range, 400)
    y = np.linspace(*y_range, 400)
    X, Y = np.meshgrid(x, y)
    Z = func.evaluate(X, Y)
    
    # Loop over frames
    images = []
    for i in range(frames):
        ax.clear()
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        
        # Plot optimizers
        for optimizer in optimizers:
            history = optimizer.simulate(func, iterations = frames)
            hx, hy = zip(*history[:i+1])
            plt.plot(hx, hy, marker='o', markersize=5, label=optimizer.label)
            
        plt.title(f'{func.label}: Step {i}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(*x_range)
        plt.ylim(*y_range)
        plt.legend(loc='upper left')
        # plt.colorbar()
        plt.savefig(f'frame_{i}.png')
        # plt.close()
        images.append(imageio.imread(f'frame_{i}.png'))

    # Save the frames as a gif
    if filename is None:
        filename = func.label + ' ' + optimizer.label + '.gif'
    imageio.mimsave( os.path.join('gifs', filename), images, fps=10)

    # Cleanup temporary files
    for i in range(frames):
        os.remove(f'frame_{i}.png')
    return

