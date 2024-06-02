# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:46:24 2024

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
        opt.Adagrad(label='Adagrad', learning_rate = 0.5),
    ]

# %%
func = cf.Quadratic()
create_pic(func, optimizers, filename = 'quadratic_Adagrad.jpg')
# create_gif(func, optimizers, filename = 'quadratic_Adagrad.gif')

# %%
func = cf.Exponential()
create_pic(func, optimizers, filename = 'exponential_Adagrad.jpg')
# create_gif(func, optimizers, filename = 'exponential_Adagrad.gif')