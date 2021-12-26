# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:53:50 2021
GAUSSIAN RANDOM FIELD GENERATOR USING CHOLESKY DECOMPOSITION
@author: Pablo Merchán-Rivera
"""

import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt


# %% Random field


def random_field(Nx, Ny, mu, sigma, corr_x=1, corr_y=1, cov_function='exponential', verbose=False):
    
    
    print('Creating random field (%dx%d)... ' % (Nx, Ny))
    
    x_min, x_max, y_min, y_max = 0.0, 1*Nx, 0.0, 1*Ny
    mesh_size_x, mesh_size_y = Nx, Ny

    start_x, stop_x, step_x = x_min + x_max/(2*mesh_size_x), x_max, (x_max - x_min)/mesh_size_x
    start_y, stop_y, step_y = y_min + y_max/(2*mesh_size_y), y_max, (y_max - y_min)/mesh_size_y 
    
    x_coord = np.arange(start_x, stop_x, step_x)
    y_coord = np.arange(start_y, stop_y, step_y)
    
    if verbose == True:
        message_random_field(start_x, stop_x, step_x, x_coord, start_y, stop_y, step_y, y_coord)
    
    mesh_coord = [None] * (Nx*Ny)
    ii = 0
    for i in range(mesh_size_x):
        for j in range(mesh_size_y):
            mesh_coord[ii] = x_coord[i], y_coord[j]
            ii += 1

    C1 = np.zeros(shape=(Nx*Ny, Nx*Ny))

    for i in range(Nx*Ny):
        for j in range(Nx*Ny):
            C1[i, j] = c1(mesh_coord[i], mesh_coord[j], corr_x, corr_y)
        C1[i, i] += 1E-7

    # perform Cholesky decomposition
    L1 = np.linalg.cholesky(C1)
      
    print('Creating gaussian random field (mu=%.3f, sigma=%.3f)' % (mu, sigma))
    
    dist = cp.Normal(0, sigma).sample((Nx*Ny))
    np.random.shuffle(dist)
    G1 = mu + np.matmul(L1, dist)
    foo = np.reshape(G1, (Nx,Ny)).T

    return foo



# %% Covariance functions


# exponential 
def c1(x, y, corr_x, corr_y):
    d = (((x[0] - y[0])/corr_x) * ((x[0] - y[0])/corr_x)) + (((x[1] - y[1])/corr_y) * ((x[1] - y[1])/corr_y))
    return np.exp(-(np.sqrt(d)))



# %% Verbose message


def message_random_field(start_x, stop_x, step_x, x_coord, start_y, stop_y, step_y, y_coord):
    print('Mesh for random field:')
    print('x-coord -->  start: %.3f, stop: %.3f, step: %.3f, length:%.3f' % (start_x, stop_x, step_x, len(x_coord)))
    print('y-coord -->  start: %.3f, stop: %.3f, step: %.3f, length:%.3f' % (start_y, stop_y, step_y, len(y_coord)))
    print('---'*10)
    print('x-coord array --> %s' % (x_coord))
    print('y-coord array --> %s' % (y_coord))
    
  


'''
# Example:

Nx = 100
Ny = 100

# hydraulic conductivity field

np.random.seed(seed=333) # random seed
mu, sigma = -3, .5  # mean and standard deviation
dist_type = 'Gaussian'  # distribution type
corr_x = 10  # correlation in x
corr_y = 10  # correlation in y
cov_function = 'exponential'  # covariance function type (exponential, squared exponential or both)

# compute field
field = random_field(Nx, Ny, mu, sigma, dist_type, corr_x, corr_y, cov_function, verbose=False)

# plt results
plt.figure('0')
plt.imshow(field, cmap='rainbow')
plt.colorbar()
plt.clim(-4.5,-1)
'''

