# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:15:14 2024

@author: afisher
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import Utils.cost_functions as cf
import Utils.optimizers as opt
from Utils.generate_figures import create_pic, create_gif
plt.close('all')

# %%

optimizers = [
        opt.Momentum(label='Momentum (0.0)', momentum=0),
        opt.Momentum(label='Momentum (0.6)', momentum=0.6),
        opt.Momentum(label='Momentum (0.9)', momentum=0.9),
    ]

# %%
func = cf.Quadratic()
create_pic(func, optimizers, filename = 'quadratic_momentum.jpg')
# create_gif(func, optimizers, filename = 'quadratic_momentum.gif')

# %%
func = cf.Exponential()
create_pic(func, optimizers, filename = 'exponential_momentum.jpg')
# create_gif(func, optimizers, filename = 'exponential_momentum.gif')

# %%
func = cf.Rosenbrock(scale=0.01)
create_pic(func, optimizers, filename = 'rosenbrock_momentum.jpg')
# create_gif(func, optimizers, filename = 'rosenbrock_momentum.gif')

# %%
func = cf.Ackley(scale=10)
create_pic(func, optimizers, filename = 'ackley_momentum.jpg')
# create_gif(func, optimizers, filename = 'ackley_momentum.gif')