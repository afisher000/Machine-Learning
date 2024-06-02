# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:51:32 2024

@author: afisher
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:38:03 2024

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
        opt.NAG(label='NAG (0.6)', momentum=0.6),
        opt.Momentum(label='Momentum (0.6)', momentum=0.6),
        opt.Adam(label='Adam', learning_rate=0.1, beta1 = 0.9, beta2=0.95)
    ]
# %%
func = cf.Quadratic()
create_pic(func, optimizers, filename = 'quadratic_Adam.jpg')
# create_gif(func, optimizers, filename = 'quadratic_Adam.gif')

# %%
func = cf.Exponential()
create_pic(func, optimizers, filename = 'exponential_Adam.jpg')
# create_gif(func, optimizers, filename = 'exponential_Adam.gif')

# %%
func = cf.Rosenbrock(scale=0.01)
create_pic(func, optimizers, filename = 'rosenbrock_Adam.jpg')
# create_gif(func, optimizers, filename = 'rosenbrock_Adam.gif')

# %%
func = cf.Ackley(scale=10)
create_pic(func, optimizers, filename = 'ackley_Adam.jpg')
# create_gif(func, optimizers, filename = 'ackley_Adam.gif')