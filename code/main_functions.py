"""
Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics
@author: noga mudrik
"""

"""
Imports
"""
import matplotlib
#from sklearn.metrics import r2_score
from webcolors import name_to_rgb
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from numpy.linalg import matrix_power
from scipy.linalg import expm
from math import e
from numpy.core.shape_base import stack
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import random
from pathlib import Path
import os
from tkinter.filedialog import askopenfilename
from datetime import date
import dill   
import scipy.io
import mat73
import warnings
import statsmodels as stats
from importlib import reload  
import statsmodels.stats as st
sep = os.sep
from IPython.core.display import display, HTML
from importlib import reload  
from scipy.interpolate import interp1d
from colormap import rgb2hex
from scipy import interpolate
import pylops

#%% The CODEL 

"""
Model Functions
"""


def init_mat(size_mat, r_seed = 0, dist_type = 'norm', init_params = {'loc':0,'scale':1}, normalize = False):
  """
  This is an initialization function to initialize matrices like G_i and c. 
  Inputs:
    size_mat    = 2-element tuple or list, describing the shape of the mat
    r_seed      = random seed (should be integer)
    dist_type   = distribution type for initialization; can be 'norm' (normal dist), 'uni' (uniform dist),'inti', 'sprase', 'regional'
    init_params = a dictionary with params for initialization. The keys depends on 'dist_type'.
                  keys for norm -> ['loc','scale']
                  keys for inti and uni -> ['low','high']
                  keys for sparse -> ['k'] -> number of non-zeros in each row
                  keys for regional -> ['k'] -> repeats of the sub-dynamics allocations
    normalize   = whether to normalize the matrix
  Output:
      the random matrix with size 'size_mat'
  """
  np.random.seed(r_seed)
  random.seed(r_seed)
  if dist_type == 'norm':
    rand_mat = np.random.normal(loc=init_params['loc'],scale = init_params['scale'], size= size_mat)
  elif dist_type == 'uni':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
        raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.uniform(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'inti':
    if 'high' not in init_params.keys() or  'low' not in init_params.keys():
      raise KeyError('Initialization did not work since low or high boundries were not set')
    rand_mat = np.random.randint(init_params['low'],init_params['high'], size= size_mat)
  elif dist_type == 'sparse':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set')

    k=init_params['k']
    b1 = [random.sample(list(np.arange(size_mat[0])),np.random.randint(1,np.min([size_mat[0],k]))) for i in range(size_mat[1])]
    b2 = [[i]*len(el) for i,el in enumerate(b1)]
    rand_mat = np.zeros((size_mat[0], size_mat[1]))
    rand_mat[np.hstack(b1), np.hstack(b2)] = 1
  elif dist_type == 'regional':
    if 'k' not in init_params.keys():
      raise KeyError('Initialization did not work since k was not set for regional initialization')

    k=init_params['k']
    splits = [len(split) for split in np.split(np.arange(size_mat[1]),k)]
    cur_repeats = [np.repeat(np.eye(size_mat[0]), int(np.ceil(split_len/size_mat[0])),axis = 1) for split_len in  splits]
    cur_repeats = np.hstack(cur_repeats)[:size_mat[1]]
    
    rand_mat = cur_repeats
  else:
    raise NameError('Unknown dist type!')
  if normalize:
    rand_mat = norm_mat(rand_mat)
  return rand_mat

  
def norm_mat(mat, type_norm = 'evals', to_norm = True):
  """
  This function comes to norm matrices by the highest eigen-value
  Inputs:
      mat       = the matrix to norm
      type_norm = what type of normalization to apply. Can be only 'evals' for now.
      to_norm   = whether to norm or not to.
  Output:  
      the normalized matrix
  """    
  if to_norm:
    if type_norm == 'evals':
      eigenvalues, _ =  linalg.eig(mat)
      mat = mat / np.max(np.abs(eigenvalues))
  return mat


def update_c(F, latent_dyn, 
             params_update_c = {'update_c_type':'inv','reg_term':0,'smooth_term':0, 'to_norm_fx' : False},clear_dyn = [], 
             direction = 'c2n',other_params = {'warm_start':False},random_state=0 , skip_error = False, cofficients = []):
  """  
  The function comes to update the coefficients of the sub-dynamics, {c_i}, by solving the inverse or solving lasso.
  Inputs:
      F               = list of sub-dynamics. Should be a list of k X k arrays. 
      latent_dyn      = latent_dynamics (dynamics dimensions X time)
      params_update_c = dictionary with keys:
          update_c_type  = options:
               - 'inv' (least squares)
               - 'lasso' (sklearn lasso)
               - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
               - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
               - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)       
               - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
               - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)
               
               
               - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
          reg_term       = scalar between 0 to 1, describe the reg. term on the cofficients
          smooth_term    = scalar between 0 to 1, describe the smooth term on the cofficients (c_t - c_(t-1))
      direction      = can be c2n (clean to noise) OR  n2c (noise to clean)
      other_params   = additional parameters for the lasso solver (optional)
      random_state   = random state for reproducability (optional)
      skip_error     = whether to skip an error when solving the inverse for c (optional)
      cofficients    = needed only if smooth_term > 0. This is the reference coefficients matrix to apply the constraint (c_hat_t - c_(t-1)) on. 
      
  Outputs: 
      coefficients matrix (k X T), type = np.array
      
  example:
  coeffs = update_c(np.random.rand(2,2), np.random.rand(2,15),{})
  """  

  if isinstance(latent_dyn,list):
    if len(latent_dyn) == 1: several_dyns = False
    else: several_dyns = True
  else:
    several_dyns = False
  if several_dyns: 
    n_times = latent_dyn[0].shape[1]-1
  else:
    n_times = latent_dyn.shape[1]-1
    
    
  params_update_c = {**{'update_c_type':'inv', 'smooth_term': 0, 'reg_term':0},**params_update_c}
  if len(clear_dyn) == 0:
    clear_dyn = latent_dyn
  if direction == 'n2c':
    latent_dyn, clear_dyn  =clear_dyn,  latent_dyn
  if isinstance(F,np.ndarray): F = [F]
  coeffs_list = []
  
  
  for time_point in np.arange(n_times):
    if not several_dyns:
      cur_dyn = clear_dyn[:,time_point]
      next_dyn = latent_dyn[:,time_point+1]
      total_next_dyn = next_dyn
      f_x_mat = []
      for f_i in F:
        f_x_mat.append(f_i @ cur_dyn)
      stacked_fx = np.vstack(f_x_mat).T 
      stacked_fx[stacked_fx> 10**8] = 10**8
    else: 
      total_next_dyn = []
      for dyn_num in range(len(latent_dyn)):
        cur_dyn = clear_dyn[dyn_num][:,time_point]
        next_dyn = latent_dyn[dyn_num][:,time_point+1]
        total_next_dyn.extend(next_dyn.flatten().tolist())
        f_x_mat = []
        for f_num,f_i in enumerate(F):
          f_x_mat.append(f_i @ cur_dyn)
        if dyn_num == 0:
          stacked_fx = np.vstack(f_x_mat).T 
        else:
          stacked_fx = np.vstack([stacked_fx, np.vstack(f_x_mat).T ])
        stacked_fx[stacked_fx> 10**8] = 10**8
    
      total_next_dyn = np.reshape(np.array(total_next_dyn), (-1,1))
    if len(F) == 1: stacked_fx = np.reshape(stacked_fx,[-1,1])
    if params_update_c['smooth_term'] > 0 and time_point > 0 :
        if len(cofficients) == 0:
            warnings.warn("Warning: you called the smoothing option without defining coefficients")
    if params_update_c['smooth_term'] > 0 and time_point > 0 and len(cofficients) > 0 :
        c_former = cofficients[:,time_point-1].reshape((-1,1))
        total_next_dyn_full = np.hstack([total_next_dyn, np.sqrt(params_update_c['smooth_term'])*c_former])
        stacked_fx_full = np.hstack([stacked_fx, np.sqrt(params_update_c['smooth_term'])*np.eye(len(stacked_fx))])
    else:
        total_next_dyn_full = total_next_dyn
        stacked_fx_full = stacked_fx   
    #if params_update_c['to_norm_fx']:
    #    stacked_fx_full  = stacked_fx_full#stacked_fx_full / np.linalg.norm(stacked_fx_full, axis=0)
    if params_update_c['update_c_type'] == 'inv' or (params_update_c['reg_term'] == 0 and params_update_c['smooth_term'] == 0):
      try:
          coeffs =linalg.pinv(stacked_fx_full) @ total_next_dyn_full.reshape((-1,1))
      except:
          if not skip_error:
              raise NameError('A problem in taking the inverse of fx when looking for the model coefficients')
          else:
              return np.nan*np.ones((len(F), latent_dyn.shape[1]))
    elif params_update_c['update_c_type'] == 'lasso' :
        #herehere try without warm start
      clf = linear_model.Lasso(alpha=params_update_c['reg_term'],random_state=random_state, **other_params)
      clf.fit(stacked_fx_full,total_next_dyn_full.T )     
      coeffs = np.array(clf.coef_)

    elif params_update_c['update_c_type'].lower() == 'fista' :
        Aop = pylops.MatrixMult(stacked_fx_full)
        print('fista')
        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'
        #other_params = {'':other_params[''],
        coeffs = pylops.optimization.sparsity.FISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'],eps = params_update_c['reg_term'] , threshkind =  params_update_c.get('threshkind') )[0]

    elif params_update_c['update_c_type'].lower() == 'ista' :
        print('ista')
        #herehere try without warm start
        if 'threshkind' not in params_update_c: params_update_c['threshkind'] ='soft'
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs = pylops.optimization.sparsity.ISTA(Aop, total_next_dyn_full.flatten(), niter=params_update_c['num_iters'] , 
                                                   eps = params_update_c['reg_term'],threshkind =  params_update_c.get('threshkind'))[0]
   
        
        
    elif params_update_c['update_c_type'].lower() == 'omp' :
        print('omp')
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs  = pylops.optimization.sparsity.OMP(Aop, total_next_dyn_full.flatten(), niter_outer=params_update_c['num_iters'], sigma=params_update_c['reg_term'])[0]
        
        
    elif params_update_c['update_c_type'].lower() == 'spgl1' :
        print('spgl1')
        Aop = pylops.MatrixMult(stacked_fx_full)
        coeffs = pylops.optimization.sparsity.SPGL1(Aop, total_next_dyn_full.flatten(),iter_lim = params_update_c['num_iters'],
                                                   tau = params_update_c['reg_term'])[0]
        
        
    elif params_update_c['update_c_type'].lower() == 'irls' :
        print('irls')
        Aop = pylops.MatrixMult(stacked_fx_full)
        
        #herehere try without warm start
        coeffs = pylops.optimization.sparsity.IRLS(Aop, total_next_dyn_full.flatten(),  nouter=50, espI = params_update_c['reg_term'])[0]

        
    else:
        
        
      raise NameError('Unknown update c type')
    coeffs_list.append(coeffs.flatten())
  coeffs_final = np.vstack(coeffs_list)

  return coeffs_final.T


def create_next(latent_dyn, coefficients, F,time_point, order = 1):
  """
  This function evaluate the dynamics at t+1 given the value of the dynamics at time t, the sub-dynamics, and other model parameters
  Inputs:
      latent_dyn    = the latent dynamics (can be either ground truth or estimated). [k X T]
      coefficients  = the sub-dynamics coefficients (used by the model)
      F             = a list of np.arrays, each np.array is a sub-dynamic with size kXk
      time_point    = current time point
      order         = how many time points in the future we want to estimate
  Outputs:
      k X 1 np.array describing the dynamics at time_point+1
  order 1 = only x_(t+1) is predicted using x_t. if order = k, x_(t+k) is predicted using x_t
  """
  if isinstance(F[0],list):
    F = [np.array(f_i) for f_i in F]

  if latent_dyn.shape[1] > 1:

    cur_A  = np.dstack([coefficients[i,time_point]*f_i @ latent_dyn[:, time_point] for i,f_i in enumerate(F)]).sum(2).T   
  else:

    cur_A  = np.dstack([coefficients[i,time_point]*f_i @ latent_dyn for i,f_i in enumerate(F)]).sum(2).T 
  if order > 1:
      cifi =  np.dstack([coefficients[i,time_point]*f_i for i,f_i in enumerate(F)]).sum(2).T 
      cifi_power = matrix_power(cifi,order-1)
      cur_A = cifi_power @ cur_A
  return cur_A

def create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = False,error_order = 1, weights_power = 1.2, weights = [], mute_infs = 10**50, max_inf = 10**60, bias_val = []):
    
  """
  An intermediate step for the reconstruction -
  Specifically - It calculated the error that should be taken in the GD step for updating f: 
  f - eta * output_of(create_ci_fi_xt)
  output: 
      3d array of the gradient step (unweighted): [k X k X time]
  """
  if len(bias_val) == 0: bias_val = np.zeros((latent_dyn.shape[0], 1))
  if max_inf <= mute_infs:
    raise ValueError('max_inf should be higher than mute-infs')
    
  if  error_order > 1:
    curse_dynamics = latent_dyn
    list_dyns = [curse_dynamics]; order_list = [1]

    for i in range(error_order):
      curse_dynamics =create_reco(curse_dynamics, coefficients, F)  # changeNM -  add a loop that create cuse according to order
      curse_dynamics[curse_dynamics > max_inf] = max_inf
      curse_dynamics[curse_dynamics < -max_inf] = -max_inf
      list_dyns.append(curse_dynamics); order_list.append(i+2)

    if len(weights) == 0:
      weights = np.array(order_list)[::-1]**weights_power/np.sum(np.array(order_list)**weights_power)
    if mute_infs > 0:
      to_mute = np.array([np.median((list_dyn-latent_dyn)**2) > mute_infs for list_dyn in list_dyns]      )
    else:
      to_mute = np.array([False] * len(list_dyns))
    if (to_mute == False).any():
    
      weights[to_mute] = 0

    else:
      mute_vals = np.array([np.median((list_dyn-latent_dyn)**2)  for list_dyn in list_dyns]      )
      weights[mute_vals > np.min(mute_vals)] = 0
  
    weights = weights / np.sum(weights)
    curse_dynamics = np.average(np.dstack(list_dyns), axis = 2, weights = np.array(order_list)[::-1]/np.sum(np.array(order_list)**2))
  else:
    curse_dynamics = latent_dyn

  all_grads = []
  for time_point in np.arange(latent_dyn.shape[1]-1):
    if cumulative:
      if time_point > 0:
        previous_A = cur_A
      else:
        previous_A = curse_dynamics[:,0]
      cur_A = create_next(np.reshape(previous_A,[-1,1]), coefficients, F,time_point)
    else:
      cur_A = create_next(curse_dynamics, coefficients, F,time_point)
    next_A = latent_dyn[:,time_point+1]
    
    """
    The actual step
    """
    if np.sum(bias_val) !=0:
        cur_A = cur_A + bias_val
    if cumulative:
      gradient_val = (next_A - cur_A) @ previous_A.T
    else:
      gradient_val = (next_A - cur_A) @ curse_dynamics[:, time_point].T
    all_grads.append(gradient_val)
  return np.dstack(all_grads)

def norm_mat_by_max(mat, to_norm = True, type_norm = 'exp'):
  """
  normalize a matrix by dividing by its max value or by fixing the determinant to 1
  """  
  if to_norm:
    if type_norm == 'max':
      mat = mat / np.max(np.abs(mat))
    elif type_norm  == 'exp':
      mat = np.exp(-np.trace(mat))*expm(mat)
  return mat

def update_f_all(latent_dyn,F,coefficients,step_f, normalize = False, acumulated_error = False,error_order = 1,action_along_time = 'mean', weights_power = 1.2, weights = [], normalize_eig = True,  bias_val = []):
    
  """
  Update all the sub-dynamics {f_i} using GD
  """
  if len(bias_val) == 0:
      bias_val = np.zeros((latent_dyn.shape[0], 1))
      
  if action_along_time == 'mean':
    if acumulated_error:
      all_grads = create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = acumulated_error, error_order = error_order, weights_power=weights_power,weights =weights, bias_val = bias_val)
      new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.mean(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
    
    else:
      all_grads = create_ci_fi_xt(latent_dyn,F,coefficients,error_order = error_order, weights_power=weights_power,weights =weights, bias_val = bias_val)
      new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.mean(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
  elif action_along_time == 'median':
    if acumulated_error:
      all_grads = create_ci_fi_xt(latent_dyn,F,coefficients, cumulative = acumulated_error, error_order = error_order, weights_power=weights_power,weights =weights, bias_val = bias_val)
      new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.median(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
    
    else:
      all_grads = create_ci_fi_xt(latent_dyn,F,coefficients,error_order = error_order, weights_power=weights_power,weights =weights, bias_val = bias_val)
      
      
      new_f_s = [norm_mat(f_i-2*step_f*norm_mat_by_max(np.median(all_grads[:,:,:]*np.reshape(coefficients[i,:], [1,1,-1]), 2),to_norm = normalize),to_norm = normalize_eig ) for i,f_i in enumerate(F)] 
  else:
    raise NameError('Unknown action along time. Should be mean or median')
  for f_num in range(len(new_f_s)):
      rand_mat = np.random.rand(new_f_s[f_num].shape[0],new_f_s[f_num].shape[1])
      new_f_s[f_num][np.isnan(new_f_s[f_num])] = rand_mat[np.isnan(new_f_s[f_num])] .flatten()
      
  return new_f_s


# def update_bias(latent_dyn, F,coefficients,action_along_time= 'median' ):
#     x_t_next = latent_dyn[:,1:]    
#     cur_reco = create_reco(latent_dyn,coefficients, F, accumulation = False, step_n = 1,type_find = action_along_time)
#     x_t_next_predicted = cur_reco[:,1:]
#     if action_along_time == 'median':
#         bias_val = np.median(x_t_next - x_t_next_predicted, 1).reshape((-1,1))
#     elif action_along_time == 'mean':
#         bias_val = np.mean(x_t_next - x_t_next_predicted, 1).reshape((-1,1))
#     else:
#         raise NameError('Unknown function along time')
#     return bias_val 
    
    
# def update_bias_out(latent_dyn, data_spec , D,action_along_time= 'median' ):
#     if along_time == 'median':
#         bias_out_val = np.median(data_spec - D @ latent_dyn, 1).reshape((-1,1))
#     elif along_time == 'mean':
#         bias_out_val = np.mean(data_spec - D @ latent_dyn, 1).reshape((-1,1))
#     return bias_out_val        
    
    
    
  
    
def lorenz(x, y, z, s=10, r=25, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def create_lorenz_mat(t = [], initial_conds = (0., 1., 1.05) , txy = []):
  """
  Create the lorenz dynamics
  """
  if len(t) == 0: t = np.arange(0,1000,0.01)
  if len(txy) == 0: txy = t
  # Need one more for the initial values
  xs = np.empty(len(t)-1)
  ys = np.empty(len(t)-1)
  zs = np.empty(len(t)-1)

  # Set initial values
  xs[0], ys[0], zs[0] = initial_conds

  # Step through "time", calculating the partial derivatives at the current point
  # and using them to estimate the next point

  for i in range(len(t[:-2])):
      dt_z = t[i+1] - t[i]
      dt_xy =  txy[i+1] - txy[i]
      x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + (x_dot * dt_xy)
      ys[i + 1] = ys[i] + (y_dot * dt_xy)
      zs[i + 1] = zs[i] + (z_dot * dt_z)
  return xs, ys, zs




def create_dynamics(type_dyn = 'cyl', max_time = 1000, dt = 0.01, change_speed = False, t_speed = np.exp, axis_speed = [],params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}):
  """
  Create ground truth dynamics
  """
  t = np.arange(0, max_time, dt)
  if type_dyn == 'cyl':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    z = t     + params_ex['bias']

    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t)
      if 0 in axis_speed: x = np.sin(t_speed_vec)
      if 1 in axis_speed: y = np.cos(t_speed_vec)
      if 2 in axis_speed: z = t_speed_vec
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'spiral':
    x = t*np.sin(t)
    y = t*np.cos(t)
    z = t 
    if change_speed: 
      t_speed_vec = t_speed(params_ex['exp_power']*t)
      if 0 in axis_speed: x = t_speed_vec * np.sin(t_speed_vec)
      if 1 in axis_speed: y = t_speed_vec * np.cos(t_speed_vec)
      if 2 in axis_speed: z = t_speed_vec
      
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'lorenz':    
    txy = t
    if change_speed: 
      #t_speed_vec = t_speed(params['exp_power']*t)
      t_speed_vec = t**params_ex['exp_power']
      if (0 and 1) in axis_speed: txy = t_speed_vec      
      if 2 in axis_speed: txy = t_speed_vec
    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'torus':
    R=5;    r=1;
    u=np.arange(0,max_time,dt);
    v=np.arange(0,max_time,dt);
    [u,v]=np.meshgrid(u,v);
    x=(R+r*np.cos(v)) @ np.cos(u);
    y=(R+r*np.cos(v)) @ np.sin(u);
    z=r*np.sin(v);
    dynamics = np.vstack([x.flatten(),y.flatten(),z.flatten()]) 
  elif type_dyn == 'circ2d':
    x = params_ex['radius']*np.sin(t)
    y = params_ex['radius']*np.cos(t)
    dynamics = np.vstack([x.flatten(),y.flatten()]) 
  elif type_dyn == 'multi_cyl':
    dynamics0 = create_dynamics('cyl',max_time = 50,params_ex = params_ex)
    list_dyns = []
    for dyn_num in range(params_ex['num_cyls']):
        np.random.seed(dyn_num)
        random_trans = np.random.rand(dynamics0.shape[0],dynamics0.shape[0])-0.5
        transformed_dyn = random_trans @ dynamics0
        list_dyns.append(transformed_dyn)
    dynamics = np.hstack(list_dyns)
  elif type_dyn == 'c_elegans':
      mat_c_elegans = load_mat_file('WT_NoStim.mat','E:\CoDyS-Python-rep-\other_models') # 
      dynamics = mat_c_elegans['WT_NoStim']['traces'].T
  elif type_dyn == 'lorenz_2d':
    txy = t
    if change_speed: 
      #t_speed_vec = t_speed(params['exp_power']*t)
      t_speed_vec = t**params_ex['exp_power']
      if (0 and 1) in axis_speed: txy = t_speed_vec      
      if 2 in axis_speed: txy = t_speed_vec
    x,y,z  = create_lorenz_mat(t, txy = txy)
    dynamics = np.vstack([x.flatten(),z.flatten()]) 
  elif type_dyn.lower() == 'fhn':
    v_full, w_full = create_FHN(dt = dt, max_t = max_time, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0, params = {'exp_power' : params_ex['exp_power'], 'change_speed': change_speed})      
    
    dynamics = np.vstack([v_full, w_full])
  return    dynamics

def rgb_to_hex(rgb_vec):
  r = rgb_vec[0]; g = rgb_vec[1]; b = rgb_vec[2]
  return rgb2hex(int(255*r), int(255*g), int(255*b))

def quiver_plot(sub_dyn = [], xmin = -5, xmax = 5, ymin = -5, ymax = 5, ax = [], chosen_color = 'red',
                alpha = 0.4, w = 0.02, type_plot = 'quiver', zmin = -5, zmax = 5, cons_color = False,
                return_artist = False,xlabel = 'x',ylabel = 'y',quiver_3d = False,inter=2):
    """
    type_plot - can be either quiver or streamplot
    """
    
    if len(sub_dyn) == 0:
        sub_dyn =  np.array([[0,-1],[1,0]])

    
    if ymin >= ymax:
        raise ValueError('ymin should be < ymax')
    elif xmin >=xmax:            
        raise ValueError('xmin should be < xmax')
    else:

        if not quiver_3d:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots()
            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin,ymax))

            new_mat = sub_dyn - np.eye(len(sub_dyn))

            U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
            V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])

            if type_plot == 'quiver':
                h = ax.quiver(X,Y,U,V, color = chosen_color, alpha = alpha, width = w)
            elif type_plot == 'streamplot':

                
                x = np.linspace(xmin,xmax,100)
                y = np.linspace(ymin,ymax,100)
                X, Y = np.meshgrid(x, y)
                new_mat = sub_dyn - np.eye(len(sub_dyn))
                U = new_mat[0,:] @ np.vstack([X.flatten(), Y.flatten()])
                V = new_mat[1,:] @ np.vstack([X.flatten(), Y.flatten()])
                

                if cons_color:

                    if len(chosen_color[:]) == 3 and isinstance(chosen_color, (list,np.ndarray)): 
                        color_stream = rgb_to_hex(chosen_color)
                    elif isinstance(chosen_color, str) and chosen_color[0] != '#':
                        color_stream = list(name_to_rgb(chosen_color))
                    else:
                        color_stream = chosen_color

                else:
                    new_mat_color = np.abs(new_mat  @ np.vstack([x.flatten(), y.flatten()]))
                    color_stream = new_mat_color.T @ new_mat_color
                try:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = color_stream) #chosen_color
                except:
                    h = ax.streamplot(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100),U.reshape(X.shape),V.reshape(Y.shape), color = chosen_color) #chosen_color
            else:
                raise NameError('Wrong plot name')
        else:
            if isinstance(ax,list) and len(ax) == 0:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
            X, Y , Z = np.meshgrid(np.arange(xmin, xmax,inter), np.arange(ymin,ymax,inter), np.arange(zmin,zmax,inter))

            new_mat = sub_dyn - np.eye(len(sub_dyn))
            U = np.zeros(X.shape); V = np.zeros(X.shape); W = np.zeros(X.shape); 

            for xloc in np.arange(X.shape[0]):
                for yloc in np.arange(X.shape[1]):
                    for zloc in np.arange(X.shape[2]):
                        U[xloc,yloc,zloc] = new_mat[0,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        V[xloc,yloc,zloc] = new_mat[1,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))
                        W[xloc,yloc,zloc] = new_mat[2,:] @ np.array([X[xloc,yloc,zloc] ,Y[xloc,yloc,zloc] ,Z[xloc,yloc,zloc] ]).reshape((-1,1))

            if type_plot == 'quiver':                    
                h = ax.quiver(X,Y,Z,U,V,W, color = chosen_color, alpha = alpha,lw = 1.5, length=0.8, normalize=True,arrow_length_ratio=0.5)#, width = w
                ax.grid(False)
            elif type_plot == 'streamplot':
                raise NameError('streamplot is not accepted for the 3d case')
         
            else:
                raise NameError('Wront plot name')
    if quiver_3d: zlabel ='z'
    else: zlabel = None
 
    add_labels(ax, zlabel = zlabel, xlabel = xlabel, ylabel = ylabel) 
    if return_artist: return h
            
            
            
    
# def create_syn(F = [], coeffs = [], n_times = 1000, dims = 2)    :
#     if isinstance(F, list) and len(F) == 0:
#         F = [np.array([[0,-1],[1,0]])]
#     if isinstance(coeffs, list) and len(coeffs) == 0:
#         coeffs = np.sin(np.linspace(0,100,n_times)).reshape((1,-1))# np.random.rand(1,n_times)# np.ones((2,n_times))
#     latent_dyn_0 = 20*np.ones(dims)
#     latent_dyn_all = np.zeros((dims,n_times))
#     latent_dyn_all[:,0] = [20,-20]#latent_dyn_0
#     dyn = create_reco(latent_dyn_all,coeffs, F, accumulation = True)
 
#     return dyn

    
    
# def movmfunc(func, mat, window = 3, direction = 0):
#   """
#   moving window with applying the function func on the matrix 'mat' towrads the direction 'direction'
#   """
#   if len(mat.shape) == 1: 
#       mat = mat.reshape((-1,1))
#       direction = 0
#   addition = int(np.ceil((window-1)/2))
#   if direction == 0:
#     mat_wrap = np.vstack([np.nan*np.ones((addition,np.shape(mat)[1])), mat, np.nan*np.ones((addition,np.shape(mat)[1]))])
#     movefunc_res = np.vstack([func(mat_wrap[i-addition:i+addition,:],axis = direction) for i in range(addition, np.shape(mat_wrap)[0]-addition)])
#   elif direction == 1:
#     mat_wrap = np.hstack([np.nan*np.ones((np.shape(mat)[0],addition)), mat, np.nan*np.ones((np.shape(mat)[0],addition))])
#     movefunc_res = np.vstack([func(mat_wrap[:,i-addition:i+addition],axis = direction) for i in range(addition, np.shape(mat_wrap)[1]-addition)]).T
#   return movefunc_res

def create_reco(latent_dyn,coefficients, F, accumulation = False, step_n = 1,type_find = 'median',min_far =10, smooth_coeffs = False, smoothing_params = {'wind':5},enable_history = True, bias_type = 'disable', bias_val = []):
  """
  This function creates the reconstruction 
  step_n: if accumulation -> how many previous samples to consider
          if accumulation == False -> the reconstruction order
  bias_type: can be:
      disable - no internal bias
      shift  - shift of the reconstructed dynamics by a fixed value
      each   - add the bias inside the reconstruction
  """
  if smooth_coeffs:
    coefficients = movmfunc(np.nanmedian, coefficients, window = smoothing_params['wind'], direction = 1)
  if accumulation:
    calcul_history = False
    cur_reco = latent_dyn[:,0].reshape((-1,1))
    for time_point in range(latent_dyn.shape[1]-1):
      next_dyn1 = create_next(cur_reco, coefficients, F,time_point)
      if step_n == 1:
        next_dyn = next_dyn1
      else:
        if (next_dyn1 < min_far).all():
          next_dyns = [next_dyn1]
        else:
          next_dyns = []

        for order in range(2,step_n+1):
          if time_point-order+1 >= 0:#cur_reco.shape[1]
            cur_next_dyn = create_next(latent_dyn, coefficients, F,time_point-order+1, order = order)
            if (cur_next_dyn < min_far).all():
              next_dyns.append(cur_next_dyn)
        if len(next_dyns) > 0:          
          if type_find == 'mean':
            next_dyn = np.dstack(next_dyns).mean(2)
          elif type_find == 'median':
            next_dyn = np.median(np.dstack(next_dyns),2)
          else:
            raise NameError('Unknown type find')
        else:
          calcul_history = True
      if enable_history and (((step_n == 1) and (not (next_dyn1 < min_far).all())) or (calcul_history)):
        addi = 1    
        while not (next_dyn < min_far).all():          
          if time_point-step_n+1-addi <=0:
            next_dyn = next_dyn1
            break
          next_dyn = create_next(latent_dyn, coefficients, F,time_point-step_n+1-addi, order = step_n+addi)# create_reco(latent_dyn,coefficients, F, accumulation, step_n = step_n+1,type_find = 'median',min_far =10, smooth_coeffs = False, smoothing_params = {'wind':5},enable_history = True)[:,-1]
          addi += 1
      else:
        next_dyn = next_dyn1
      if bias_type == 'each'  and len(bias_val) > 0:
          cur_reco = np.hstack([cur_reco, next_dyn.reshape(-1,1) + bias_val.reshape(-1,1)])
      else:    
          cur_reco = np.hstack([cur_reco, next_dyn.reshape(-1,1)])
  else:
    if bias_type == 'each'  and len(bias_val) > 0:
        cur_reco = np.hstack([create_next(latent_dyn, coefficients, F,time_point)+ bias_val.reshape(-1,1) for time_point in range(latent_dyn.shape[1]-1)])
        cur_reco = np.hstack([latent_dyn[:,0].reshape((-1,1)),cur_reco])
    else:
        cur_reco = np.hstack([create_next(latent_dyn, coefficients, F,time_point) for time_point in range(latent_dyn.shape[1]-1)])
        cur_reco = np.hstack([latent_dyn[:,0].reshape((-1,1)),cur_reco])
    
    if step_n <= 1:
        pass
    else:
      cur_reco = create_reco(cur_reco,coefficients, F, accumulation = False, step_n = step_n-1,type_find = type_find, smooth_coeffs = smooth_coeffs, smoothing_params = smoothing_params)
  if bias_type == 'shift' and len(bias_val) > 0:
      cur_reco = cur_reco + bias_val.reshape(-1,1)
  return cur_reco

#%% Post-Proc Functions
  
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='Z', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {},zlabel_params = {}, xlabel_params = {},  title_params = {}):
  """
  This function add labels, titles, limits, etc. to figures;
  Inputs:
      ax      = the subplot to edit
      xlabel  = xlabel
      ylabel  = ylabel
      zlabel  = zlabel (if the figure is 2d please define zlabel = None)
      etc.
  """
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**ylabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)



def visualize_dyn(dyn,ax = [], params_plot = {},turn_off_back = False, marker_size = 10, include_line = False, 
                  color_sig = [],cmap = 'cool', return_fig = False, color_by_dominant = False, coefficients =[],
                  figsize = (5,5),colorbar = False, colors = [],vmin = None,vmax = None, color_mix = False, alpha = 0.4,
                  colors_dyns = np.array(['r','g','b','yellow']), add_text = 't ', text_points = [],fontsize_times = 18, 
                  marker = "o",delta_text = 0.5, color_for_0 =None, legend = [],fig = [],return_mappable = False):
   """
   Plot the multi-dimensional dynamics
   Inputs:
       dyn          = dynamics to plot. Should be a np.array with size k X T
       ax           = the subplot to plot in (optional)
       params_plot  = additional parameters for the plotting (optional). Can include plotting-related keys like xlabel, ylabel, title, etc.
       turn_off_back= disable backgroud of the plot? (optional). Boolean
       marker_size  = marker size of the plot (optional). Integer
       include_line = add a curve to the plot (in addition to the scatter plot). Boolean
       color_sig    = the color signal. if empty and color_by_dominant - color by the dominant dynamics. If empty and not color_by_dominant - color by time.
       cmap         = cmap
       colors       = if not empty -> pre-defined colors for the different sub-dynamics. Otherwise - colors are according to the cmap.
       color_mix    = relevant only if  color_by_dominant. In this case the colors need to be in the form of [r,g,b]
   Output:
       (only if return_fig) -> returns the figure      
      
   """
   if not isinstance(color_sig,list) and not isinstance(color_sig,np.ndarray): color_sig = [color_sig]

   #display(ax)

   if isinstance(ax,list) and len(ax) == 0:
       if dyn.shape[0] == 3:
           fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
       else:
           fig, ax = plt.subplots(figsize = figsize)  
           
       

   if include_line:
       if dyn.shape[0] == 3:
           ax.plot(dyn[0,:], dyn[1,:], dyn[2,:],alpha = 0.2)
       else:
           ax.plot(dyn[0,:], dyn[1,:], alpha = 0.2)
   if len(legend) > 0:
       [ax.scatter([],[], c = colors_dyns[i], label = legend[i], s = 10) for i in np.arange(len(legend))]
       ax.legend()
   # Create color sig        
   if len(color_sig) == 0: 
       color_sig = np.arange(dyn.shape[1])      
   if color_by_dominant and (coefficients.shape[1] == dyn.shape[1]-1 or coefficients.shape[1] == dyn.shape[1]): 
       if color_mix:
           if len(colors) == 0 or not np.shape(colors)[0] == 3: raise ValueError('colors mat should have 3 rows')
           else:
               #color_sig = ((colors[:,:coefficients.shape[0]] @ coefficients)  / coefficients.sum(0).reshape((1,-1))).T
               #color_sig = ((np.array(colors)[:,:coefficients.shape[0]] @ np.abs(coefficients))  / np.abs(coefficients).sum(0).reshape((1,-1))).T
               color_sig = ((np.array(colors)[:,:coefficients.shape[0]] @ np.abs(coefficients))  / np.max(np.abs(coefficients).sum(0).reshape((1,-1)))).T
               color_sig[np.isnan(color_sig) ] = 0.1
               #display(color_sig.max())
               dyn = dyn[:,:-1]
       else:
           
           color_sig_tmp = find_dominant_dyn(coefficients)
           if len(colors_dyns) > 0: 
               color_sig = colors_dyns[color_sig_tmp]
           elif len(color_sig) == 0:  
               color_sig=color_sig_tmp 
           else:        
               color_sig=np.array(color_sig)[color_sig_tmp] 
           if len(color_sig.flatten()) < dyn.shape[1]: dyn = dyn[:,:len(color_sig.flatten())]
           if color_for_0:

               color_sig[np.sum(coefficients,0) == 0] = color_for_0
           #if len(colors) > 0:
           #    color_sig = colors[:,color_sig.flatten()].T
   # The actual plotting

   if dyn.shape[0] > 2:
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker = marker, s = marker_size,c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:], dyn[2,:], marker =marker, s = marker_size,c= color_sig, alpha = alpha)
   else:
       dyn = np.array(dyn)
       
       if len(colors) == 0:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig,cmap = cmap, alpha = alpha,
                          vmin = vmin, vmax = vmax)
       else:
           h = ax.scatter(dyn[0,:], dyn[1,:],  marker = marker, s = marker_size,c= color_sig, alpha = alpha)
  
           params_plot['zlabel'] = None
   if len(params_plot) > 0:
     if dyn.shape[0] == 3:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
     else:
         if 'xlabel' in params_plot.keys():
           add_labels(ax, xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         elif 'zlabel' in params_plot.keys():
               add_labels(ax,  zlabel=None, title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None)
         else:
           add_labels(ax,   title=params_plot.get('title'),
                     xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =None,zlabel = None);
   if len(text_points) > 0:
       
       if dyn.shape[0] == 3:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text,dyn[2,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]
       else:
           [ax.text(dyn[0,t]+delta_text,dyn[1,t]+delta_text, '%s = %s'%(add_text, str(t)),  fontsize =fontsize_times, fontweight = 'bold') for t in text_points]

   remove_edges(ax)
   ax.set_axis_off()
   if colorbar:
       fig.colorbar(h, cax=ax, position = 'top')
   if return_mappable:
       return h
       
   
    
  # else:
  #     if dyn.shape[0] > 2:    add_labels(ax)
  #     else: add_labels(ax, zlabel = None)
      
  # if turn_off_back and dyn.shape[0] == 3:
  #   ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  #   ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  #   ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  #   ax.grid(False)
  # if colorbar: 
  #     if color_by_dominant:  
  #         cbar = plt.colorbar(h, ticks=np.arange(coefficients.shape[0])+0.5) 
  #         cbar.ax.set_yticklabels(['c%g'%i for i in range(coefficients.shape[0])])
  #     else:
  #         plt.colorbar(h)
  # if return_fig:
  #     return h

def find_dominant_dyn(coefficients):
    """
    This function finds the # of the most dominant dynamics in each time point. It should be used when comparing to rsLDS
    Input:
        coefficients: np.array of kXT
    Output:
        an array with len T, containing the index of the most dominant sub-dynamic at each time point
    """
    domi = np.argmax(np.abs(coefficients),0)
    return domi  


# def check_subs_effect(latent_dyn,F,coefficients, ax = [], dict_store = {}, pre_name ='without', to_plot = True , min_time = 0, params_plot = {}, update_coeffs = True,  
#                       color_sig_type = 'mse', fig = [], title_fig = '', include_colorbar = False,cmap = 'cool', random_colors = True,store_data = True,
#                       plot_percent = False,range_close = [],ax_percent = [], plot_backward = True, plot_forward = True, figsize = (15,10)):
#   """
#   Check the effect of each sub-dynamics by exploring the gain in error when removing it, and the gain of error when using only it. 
#   """    

#   if len(range_close) == 0: range_close = np.linspace(10**-8, 10,30)
#   num_subdyns = len(F)
#   withouts = [list(itertools.combinations(np.arange(num_subdyns),k)) for k in range(num_subdyns)]  
#   colors = np.random.rand(3,coefficients.shape[0])
#   if store_data: 
#       stored_contri = {'Gain with':pd.DataFrame(np.zeros((coefficients.shape[0],2)), index = np.arange(coefficients.shape[0]),columns= ['1-error','% correct']),'Loss without':pd.DataFrame(np.zeros((coefficients.shape[0],2)), index = np.arange(coefficients.shape[0]), columns = ['error','% wrong'])}
#   if plot_percent:
#       if isinstance(ax_percent,list):
#         if len(ax_percent) == 0:
#             fig_percent, ax_percent = plt.subplots(2,2, figsize =figsize)      
#   if to_plot:
#     if isinstance(ax, list):
#       if len(ax) == 0:
#          max_len_without = np.max([len(without_spec) for without_spec in withouts])
#          if latent_dyn.shape[0] == 3:         fig, ax = plt.subplots(len(withouts),max_len_without,figsize = figsize, subplot_kw={'projection':'3d'})  
#          else:                                fig, ax = plt.subplots(len(withouts),max_len_without,figsize = figsize)  
         
#     if not isinstance(ax,np.ndarray):        ax = np.array([[ax]])
#     if len(ax.shape) == 1: ax = ax.reshape((-1,1))
#   for group_num, without_group in enumerate(withouts):
#     for without_num, without in enumerate(without_group):
#       with_subs = list(set(np.arange(num_subdyns)) - set(without))  
#       if update_coeffs:        
#         coeffs_run= update_c(np.array(F)[with_subs].tolist(),latent_dyn[:,min_time:],{})        
#       else:
#         if len(with_subs) == 1: coeffs_run = coefficients[np.array(with_subs),min_time:].reshape((1,-1))          
#         else: 
#             coeffs_run = coefficients[np.array(with_subs),min_time:]            
#       F_run = [f_i for i,f_i in enumerate(F) if i in with_subs]
#       reco = create_reco(latent_dyn[:,min_time:],coeffs_run, F_run)
#       name_store = '_'.join(['without'] + [str(num_without) for num_without in without])
#       dict_store[name_store] = reco
#       mse_without = np.sqrt(np.mean((reco-latent_dyn[:,min_time:])**2))
#       ## Store data      
#       if store_data:
#           if len(with_subs) == 1:
#               calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'forward')
#               stored_contri['Gain with'].iloc[with_subs[0],:] =  calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'forward')
#               if plot_forward:
#                   plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.05, ax =ax_percent[1,0], color =colors[:,with_subs[0]] , label = with_subs[0])
              
#           elif len(without) == 1:

#               stored_contri['Loss without'].iloc[without[0],:] =  calcul_contribution(reco, latent_dyn[:,min_time:], direction = 'backward')
#               if plot_backward:
#                   plot_dots_close(reco,latent_dyn[:,min_time:], range_close =range_close, conf_int = 0.25, ax =ax_percent[1,1], color =colors[:,without[0]] , label = without[0])          
#       ## Plot
#       if to_plot:
#         if color_sig_type == 'mse': 
#             color_sig = np.mean(np.abs(reco-latent_dyn[:,min_time:]),0)
#             color_by_dominant = False
#         elif color_sig_type == 'coeffs':
#              color_sig =with_subs# np.array(with_subs)
#              color_by_dominant = True
#         else:
#             color_sig =[]
#             color_by_dominant = False
#         h = visualize_dyn(reco, ax[group_num, without_num], params_plot, color_sig= color_sig, return_fig = True, colors_dyns = [], color_by_dominant = color_by_dominant, coefficients =coeffs_run  ,cmap = cmap, colors = [], vmin = 0, vmax = coefficients.shape[0])

#         if len(params_plot) > 0:
#           add_labels(ax[group_num, without_num], xlabel=params_plot.get('xlabel'), ylabel=params_plot.get('ylabel'), zlabel=params_plot.get('zlabel'), title=params_plot.get('title'),
#             xlim = params_plot.get('xlim'), ylim  =params_plot.get('ylim'), zlim =params_plot.get('zlim'))
#         else:

#           if latent_dyn.shape[0] == 3:
#               add_labels(ax[group_num, without_num], title = name_store + ' rMSE: '+'%g'%mse_without ,
#                          xlim = [np.min(latent_dyn[0,:])-2,np.max(latent_dyn[0,:])+2],
#                          ylim =[np.min(latent_dyn[1,:])-2,np.max(latent_dyn[1,:])+2], zlim =[np.min(latent_dyn[2,:])-2,np.max(latent_dyn[2,:])+2])
#           else:
#               add_labels(ax[group_num, without_num], title = name_store + ' rMSE: '+'%g'%mse_without ,
#                          xlim = [np.min(latent_dyn[0,:])-2,np.max(latent_dyn[0,:])+2],
#                          ylim =[np.min(latent_dyn[1,:])-2,np.max(latent_dyn[1,:])+2], zlabel = None)

#             #[ax_spec.axis('off') for ax_num,ax_spec in enumerate(ax.flatten()) if ax_num > without_num]
#     [ax_spec.axis('off') for ax_num,ax_spec in enumerate(ax[group_num, :]) if ax_num > without_num]
#   fig.suptitle(title_fig);
#   if include_colorbar:
#       fig.subplots_adjust(right=0.7)
#       cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
#       fig.colorbar(h, cax=cbar_ax)
#   if store_data: 
#       if plot_percent:
#           ax_percent[1,0].legend()
#           ax_percent[1,1].legend()
#           return stored_contri, dict_store, h, ax_percent,fig_percent
#       return stored_contri, dict_store, h
#   return dict_store, h

# def check_eigenspaces(F, colors = [],figsize = (15,8), ax = [], title2 = 'Eigenspaces of different sub-dynamics',title1= 'Eigenvalues of different sub-dynamics'):
#   fig = plt.figure(figsize = figsize)
#   ax1 = fig.add_subplot(121)
#   if np.shape(F[0])[0] == 3:
#       ax2 = fig.add_subplot(122, projection='3d')
#   else:
#       ax2 = fig.add_subplot(122)
#   if len(colors) == 0:
#     colors = np.random.rand(3,len(F))
#   evals_list = []
#   evecs_list = []
#   for f_num, f_i in enumerate(F):
#     if isinstance(colors,np.ndarray):
#       cur_color =  [list(colors[:,f_num])]
#     else:
#       cur_color = colors[f_num]
#     eigenvalues, eigenvectors =  linalg.eig(f_i)
#     evals_list.append(eigenvalues)
#     evecs_list.append(eigenvectors)
#     ax1.scatter(np.real(eigenvalues),np.imag(eigenvalues),marker = 'o', c =cur_color, label = 'G%g'%f_num)
#     eigenvectors_real = np.real(eigenvectors)

#     if eigenvectors_real.shape[0] == 3:
#         ax2.scatter( eigenvectors_real[0,:],eigenvectors_real[1,:], eigenvectors_real[2,:],marker = 'o', c = cur_color, label = 'G%g'%f_num)
#     elif eigenvectors_real.shape[0] == 2:
#         ax2.scatter( eigenvectors_real[0,:],eigenvectors_real[1,:],marker = 'o', c = cur_color, label = 'G%g'%f_num)

#     # 1. create vertices from points
#     if eigenvectors_real.shape[0] == 3:
#         verts = [list(zip(eigenvectors_real[0,:],eigenvectors_real[1,:],eigenvectors_real[2,:]))]
#     if eigenvectors_real.shape[0] == 2:
#         verts = [list(zip(eigenvectors_real[0,:],eigenvectors_real[1,:]))]
#     srf = Poly3DCollection(verts, alpha=.25, facecolor= cur_color)

#     ax2.add_collection3d(srf)
#   add_labels(ax2, title=title2)
#   add_labels(ax1, xlabel='Real', ylabel = 'Img',zlabel =None,  title=title1)

#   return evecs_list,evals_list


# def add_arrow(ax, start, end,arrowprops = {'facecolor' : 'black', 'width':1, 'alpha' :0.2} ):
#     arrowprops = {**{'facecolor' : 'black', 'width':1.5, 'alpha' :0.2, 'edgecolor':'none'}, **arrowprops}
#     ax.annotate('',ha = 'center', va = 'bottom',  xytext = start,xy =end,
#                 arrowprops = arrowprops)

    
    
def plot_sub_effect(sub_dyn, rec_rad_all = 5, colors = ['r','g','b','m'], alpha = 0.8, ax = [], 
                    n_points = 100, figsize = (10,10), params_labels = {'title':'sub-dyn effect'}, lw = 4):
    params_labels = {**{'zlabel':None}, **params_labels}
    if isinstance(ax,list) and len(ax) == 0:
        fig, ax = plt.subplots(figsize = figsize)
    if len(colors) == 1: colors = [colors]*4
    if not isinstance(rec_rad_all,list): rec_rad_all = [rec_rad_all]
    ax.axhline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.axvline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    for rec_rad in rec_rad_all:        
        ax.plot([-rec_rad, rec_rad],[rec_rad,rec_rad],alpha = alpha**2, color = colors[0], ls ='--',lw=lw)
        ax.plot([-rec_rad, rec_rad],[-rec_rad,-rec_rad],alpha = alpha**2, color = colors[1], ls = '--',lw=lw)
        ax.plot([rec_rad,  rec_rad],[-rec_rad,rec_rad],alpha = alpha**2, color = colors[2], ls = '--',lw=lw)
        ax.plot([-rec_rad,-rec_rad], [ -rec_rad,  rec_rad],alpha = alpha**2, color = colors[3], ls = '--',lw=lw)
    

        if not (sub_dyn == 0).all():
            sub_dyn = norm_mat(sub_dyn, type_norm = 'evals')
        effect_up = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [rec_rad]*n_points])
        effect_down = sub_dyn @ np.vstack([np.linspace(-rec_rad, rec_rad, n_points), [-rec_rad]*n_points])
        effect_right = sub_dyn @ np.vstack([[rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        effect_left = sub_dyn @ np.vstack([[-rec_rad]*n_points,np.linspace(-rec_rad, rec_rad, n_points)])
        ax.plot(effect_up[0,:],effect_up[1,:],alpha = alpha, color = colors[0],lw=lw)
        ax.plot(effect_down[0,:],effect_down[1,:],alpha = alpha, color = colors[1],lw=lw)
        ax.plot(effect_right[0,:],effect_right[1,:],alpha = alpha, color = colors[2],lw=lw)
        ax.plot(effect_left[0,:],effect_left[1,:],alpha = alpha, color = colors[3],lw=lw)
        # Up
        add_arrow(ax, [0,rec_rad], [np.mean(effect_up[0,:]),np.mean(effect_up[1,:])],arrowprops = {'facecolor' :colors[0]})
        add_arrow(ax, [0,-rec_rad], [np.mean(effect_down[0,:]),np.mean(effect_down[1,:])],arrowprops = {'facecolor' :colors[1]})
        add_arrow(ax, [rec_rad,0], [np.mean(effect_right[0,:]),np.mean(effect_right[1,:])],arrowprops = {'facecolor' :colors[2]})
        add_arrow(ax, [-rec_rad,0], [np.mean(effect_left[0,:]),np.mean(effect_left[1,:])],arrowprops = {'facecolor' :colors[3]})
    add_labels(ax, **params_labels)


    

def plot_evals_evecs(ax, sub_dyn, colors =['r','g','b','m'] , alpha = 0.7, title ='evals'):
    eigenvalues, eigenvectors =  linalg.eig(sub_dyn)
    for eval_num, eigenval in enumerate(eigenvalues):
        #add_arrow(ax, [0,0], [eigenvectors[0,eval_num], eigenvectors[1,eval_num]], arrowprops = {'facecolor' :colors[eval_num]})
        ax.scatter( np.real(eigenval),np.imag(eigenval), alpha = alpha, color = colors, s = 300) #[eval_num])
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.axhline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.axvline(0, alpha = 0.1, color = 'black', ls = 'dotted')
    ax.set_title('evals')
    

def plot_3d_color_scatter(latent_dyn,coefficients, ax = [], figsize = (15,10), delta = 0.4, colors = []):
    
    if latent_dyn.shape[0] != 3:
        print('Dynamics is not 3d')
        pass
    else:
        if len(colors) == 0:
            colors = ['r','g','b']
        if isinstance(ax,list) and len(ax) == 0:
            fig, ax = plt.subplots(figsize = figsize, subplot_kw={'projection':'3d'})  
        for row in range(coefficients.shape[0]):
            coefficients_row = coefficients[row]
            coefficients_row[coefficients_row == 0]  = 0.01
            
            ax.scatter(latent_dyn[0,:]+delta*row,latent_dyn[1,:]+delta*row,latent_dyn[2,:]+delta, s = coefficients_row**0.3, c = colors[row])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)

def compare_coeffs_to_discrete_coeffs(coefficients_n, axs = [],figsize = (10,15), colors = [[1, 0, 0], [0, 1, 0], [0, 0,1]],
                                      n_rep =350,type_plot = 'heatmap',
                                      titles = ['Dynamics \n Decomposition','Discrete Dynamics \n (flexible weights)','Discrete Dynamics \n (fixed weights)'],
                                      include_inter = True):
    """
    Plot the normalized coefficients of our model vs the discrete versions
    type_plot:  can be 'heatmap' or 'plot'
    """
    if isinstance(axs,list) and len(axs) == 0:
        if include_inter:        
            fig, axs = plt.subplots(3,1, sharex = True, figsize = figsize)
        else:
            fig, axs = plt.subplots(2,1, sharex = True, figsize = figsize)
    colors = np.array(colors[:coefficients_n.shape[0]])        
    
    averaged_cols = np.array(colors).T @ coefficients_n
    dstack_res = np.dstack([averaged_cols[color_num,:] for color_num in range(len(colors[0]))])
    
    max_ind = np.argmax(coefficients_n, axis = 0)
    
    # Coefficients weight fixed
    coefficients_n_zeroed = np.zeros(coefficients_n.shape)
    for max_ind_num, max_ind_spec in enumerate(max_ind): 
        coefficients_n_zeroed[max_ind_spec,max_ind_num]  = coefficients_n[max_ind_spec,max_ind_num]
    averaged_cols_zeroed = np.array(colors).T @ coefficients_n_zeroed
    dstack_res_zeroed = np.dstack([averaged_cols_zeroed[color_num,:] for color_num in range(len(colors[0]))])
    if type_plot == 'heatmap':
        dstack_res_zeroed = dstack_res_zeroed / np.max(dstack_res_zeroed, 1)
        axs[0].imshow(np.repeat(dstack_res[:,:],n_rep, axis = 0), alpha = 0.5)
        if include_inter:
            axs[1].imshow(np.repeat(dstack_res_zeroed[:,:],350, axis = 0), alpha = 0.5)
    elif type_plot == 'plot':
        
        [axs[0].plot(coefficients_n[i,:], color = colors[i]) for i in range(coefficients_n.shape[0])];    
        if include_inter:
            [axs[1].plot(coefficients_n_zeroed[i,:], color = colors[i]) for i in range(coefficients_n_zeroed.shape[0])];
    else:
        raise NameError('Type plot value is invalid. Should be "heatmap" or "plot"')
    # Coefficients weight fixed
    coefficients_n_zeroed_2 = np.zeros(coefficients_n.shape)
    for max_ind_num, max_ind_spec in enumerate(max_ind): 
        coefficients_n_zeroed_2[max_ind_spec,max_ind_num]  = 1# coefficients_n_zeroed[max_ind_spec,max_ind_num]
    averaged_cols_zeroed = np.array(colors).T @ coefficients_n_zeroed_2
    dstack_res_zeroed = np.dstack([averaged_cols_zeroed[color_num,:] for color_num in range(len(colors[0]))])
    dstack_res_zeroed = dstack_res_zeroed / np.sum(dstack_res_zeroed,2).reshape((1,-1,1))
    
    if include_inter: num_plot = 2
    else: num_plot = 1
    if type_plot == 'heatmap':

        axs[num_plot].imshow(np.repeat(dstack_res_zeroed[:,:],350, axis = 0), alpha = 0.5)
    elif type_plot == 'plot':
        [axs[num_plot].plot(coefficients_n_zeroed_2[i,:], color = colors[i]) for i in range(coefficients_n_zeroed_2.shape[0])];
    axs[-1].set_xlabel('Time')
    if type_plot == 'heatmap': [ax.set_title(titles[i]) for i, ax in enumerate(axs)]
    elif type_plot == 'plot':  [ax.set_ylabel(titles[i]) for i, ax in enumerate(axs)]
    [ax.spines['top'].set_visible(False) for ax in axs]
    [ax.spines['right'].set_visible(False) for ax in axs]
    [ax.spines['bottom'].set_visible(False) for ax in axs]
    [ax.spines['left'].set_visible(False)    for ax in axs]
    [ax.get_xaxis().set_ticks([]) for ax in axs]
    [ax.get_yaxis().set_ticks([]) for ax in axs]
    if type_plot == 'heatmap': fig.subplots_adjust(wspace=0.02,hspace=0.02)
    elif type_plot == 'plot': fig.subplots_adjust(wspace=0.3,hspace=0.3)


def add_dummy_sub_legend(ax, colors,lenf, label_base = 'f'):
    dummy_lines = []
    for i,color in enumerate(colors[:lenf]):
        dummy_lines.append(ax.plot([],[],c = color, label = '%s %s'%(label_base, str(i)))[0])
    ax.set_title('Dynamics colored by mix of colors of the dominant dynamics')
    legend = ax.legend([dummy_lines[i] for i in range(len(dummy_lines))], ['f %s'%str(i) for i in range(len(colors))], loc = 'upper left')
    ax.legend()
        
def plot_subs_effects_2d(F, colors =[['r','maroon','darkred','coral'],['forestgreen','limegreen','darkgreen','springgreen']] , alpha = 0.7 , rec_rad_all = 5, 
                         n_points = 100,  params_labels = {'title':'sub-dyn effect'}, lw = 4, evec_colors = ['r','g'], include_dyn = False, loc_leg = 'upper left', 
                         axs = [], fig = []):
    if include_dyn:
        fig, axs = plt.subplots(len(F), 3, figsize = (35,8*len(F)),sharey='col', sharex = 'col')
    else:
        if isinstance(axs,list) and len(axs) == 0:
            fig, axs = plt.subplots(len(F), 2, figsize = (30,8*len(F)),sharey='col', sharex = 'col')
            
    if isinstance(colors[0], list):
        [plot_sub_effect(f_i, rec_rad_all , colors[i] , alpha, axs[i,1], n_points, params_labels = {'title':'f %s effect'%str(i+1)}, lw = lw) for i,f_i in enumerate(F)]

    else:
        [plot_sub_effect(f_i, rec_rad_all , colors , alpha, axs[i,1], n_points, params_labels = {'title':'f %s effect'%str(i+1)}, lw = lw) for i,f_i in enumerate(F)]
        
    [plot_evals_evecs(axs[i,0], f_i, evec_colors[i] , alpha) for i,f_i in enumerate(F)]
    dummy_lines = []
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '--', lw = lw)[0])
    dummy_lines.append(axs[0,0].plot([],[], c="black", ls = '-', lw = lw)[0])
    #lines = axs.get_lines()
    legend = axs[0,1].legend([dummy_lines[i] for i in [0,1]], ['Original', 'after sub-dynamic transform'], loc = loc_leg )
    axs[0,1].add_artist(legend)
    if include_dyn:
        [quiver_plot(sub_dyn = f, ax = axs[i,2], chosen_color = evec_colors[i], type_plot='streamplot',cons_color =True, xlabel = 'dv',ylabel = 'dw') for i,f in enumerate(F)]
        [axs[i,2].set_title('f %s'%str(i+1), fontsize = 18) for i in range(len(F))]
        
    fig.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    
def plot_subs(F, axs = [],params_F_plot = {'cmap':'PiYG'}, include_sup = True,annot = True):
  """
  This function plots heatmaps of the sub-dynamics
  """
  params_F_plot = {**{'cmap':'PiYG'},**params_F_plot}
  if isinstance(axs,list):
    if len(axs) == 0:
      fig, axs = plt.subplots(1,len(F), sharex = True,sharey = True)

  [sns.heatmap(f_i, ax = axs[i],annot=annot, **params_F_plot) for i,f_i in enumerate(F)]
  [ax.set_title('f#%g'%i) for i,ax in enumerate(axs)]
  if include_sup: plt.suptitle('Sub-Dynamics')
  plt.subplots_adjust(hspace = 0.5,wspace = 0.5)


# def angle_between(A,B):
#     """
#     Calculate the angle between the matrices A and B
#     """
#     trace_AB = np.trace(A.T @ B)
#     deno = np.sqrt(np.sum(A**2)*np.sum(B**2))
#     angle = np.arccos(trace_AB/deno)
#     return angle


#%% Analysis Functions



# def check_denoising(latent_dyn,F, noise_min = 0, noise_max = 10,noise_intervals = 1, params_update_c ={}, 
#                     r_seed = 0, pre_defined = [],return_recos = True, n_samples = 5,accumulation=False,type_add = 'extend' ):
#   """
#   Check the model ability to denoise OR create noisy data
#   pre_defined = noises to add
#   """
#   np.random.seed(r_seed)
#   if len(pre_defined) == 0 and n_samples == 1:    
#     pre_defined = np.arange(noise_min, noise_max, noise_intervals)
#     noise_original = [latent_dyn+sigma*np.random.randn(latent_dyn.shape[0],latent_dyn.shape[1] )  for sigma in pre_defined]
      
#   else:
#     if len(pre_defined) == 0: pre_defined = np.arange(noise_min, noise_max, noise_intervals)
#     # if len(pre_defined) == 1: 
#     #     pre_defined = pre_defined*n_samples
#     #     change_seed = True
#     # else:
#     #     change_seed = False
#     noise_original = []
#     for noise_counter, sigma in enumerate(pre_defined):
#       noise_per_noise = []
#       for repeat in range(n_samples):        
#         np.random.seed(noise_counter*repeat)
#         noise_per_noise.append(latent_dyn+sigma*np.random.randn(latent_dyn.shape[0],latent_dyn.shape[1] )) 
#       if type_add == 'extend':
#           noise_original.extend(noise_per_noise)
#       else:
#           noise_original.append(noise_per_noise)
#   if return_recos:         
#     noise_recos = [create_reco(latent_dyn,update_c(F,noise_original[sigma_num],params_update_c),F, accumulation=accumulation) for sigma_num, sigma in enumerate(pre_defined)]  
#     return noise_recos, noise_original, np.arange(noise_min, noise_max, noise_intervals)
#   else:
#     return noise_original


# def check_speed_vary(F,dynamic_type = 'cyl',max_time = 500, dt = 0.01, change_speed = True, t_speed = np.exp, axis_speed = [0,1,2],params =  {'exp_power':0.1}):
#         speed_dyn = create_dynamics(type_dyn =dynamic_type, max_time = max_time, dt = dt, change_speed = change_speed, t_speed = t_speed, axis_speed = axis_speed,params_ex =  params)
#         #speed_dyn = create_dynamics(type_dyn = dynamic_type, max_time = max_time, dt = dt, change_speed = True, t_speed = np.exp, axis_speed = axis_speed,params = {'exp_power':exp_power})
#         new_c = update_c(F,speed_dyn,{})
#         reco_speed = create_reco(speed_dyn, new_c,F)
#         return speed_dyn, reco_speed, new_c

# def check_dist_between_subs(F, to_norm_mse = True, F_compare = [])  :
#   """
#   mse etc
#   """
#   ##
#   if len(F_compare) == 0:
#     F_compare = F
#   store_dict = {'mat':{key:np.nan*np.ones((len(F),len(F))) for key in ['rMSE','CORR','msePART','ANGLE']},
#                 'list':{key:[] for key in ['rMSE','CORR','msePART','ANGLE']} }

#   combinations = list(itertools.combinations(np.arange(len(F)), 2))
#   for comb in combinations:
#     store_dict = compute_dist_2_mat(F[comb[0]],F[comb[1]], to_norm_mse = True, rank_digits = 2, store_dict = store_dict, ind1 =comb[0] ,ind2 = comb[1])
  
#   return store_dict, combinations

# def compute_participation_factor(mat):
#   w, vl, vr = linalg.eig(mat, left=True)
#   part_factor  = np.hstack([np.abs(vl[:,w_i_num]*vr[:,w_i_num]).reshape((-1,1)) for w_i_num, w_i in enumerate(w) ])
#   return part_factor


# def compute_dist_2_mat(mat1,mat2, to_norm_mse = True, rank_digits = 4, store_dict = {}, ind1 = -1, ind2 = -1):
#   """
#   Computing the distance between a pair of matrices
#   unique_rank: the bigger this value is, the more similar the matrices are
#   """
#   to_store = (ind1 >= 0) and (ind2 >= 0)
#   metric_list = ['rMSE','CORR','msePART','ANGLE']
#   vals_dict = {metric: None for metric in metric_list}
#   if to_store and len(store_dict) == 0:
#     store_dict = {key:np.nan*np.ones((ind1,ind2)) for key in metric_list}
#   # MSE
#   if to_norm_mse:
#     mat1norm = (mat1 - np.mean(mat1))/np.std(mat1)
#     mat2norm = (mat2 - np.mean(mat2))/np.std(mat2)
#   else:
#     mat1norm = mat1; mat2norm = mat2
#   mse = (np.mean((mat1norm - mat2norm)**2))**0.5
#   if to_store: vals_dict['rMSE'] = mse
#   # Corr
#   corr_mat = np.corrcoef(mat1.flatten(),mat2.flatten())
#   corr = np.abs(corr_mat[0,1])
#   if to_store: vals_dict['CORR'] = corr
#   #angle
#   angle = angle_between(mat1,mat2)
#   if to_store: vals_dict['ANGLE'] = angle

#   part1 = compute_participation_factor(mat1) 
#   part2 = compute_participation_factor(mat2)
#   mse_part = np.mean((part1 - part2)**2)
#   if to_store: vals_dict['msePART'] = mse_part

#   if to_store: 
#     for  metric in metric_list:
#       store_dict['mat'][metric][ind1,ind2] = vals_dict[metric] 
#       store_dict['list'][metric].append(vals_dict[metric] )
#     return store_dict    
#   return mse, corr, angle, mse_part # shared_rank

# def spec_corr(v1,v2):
#   """
#   absolute value of correlation
#   """
#   corr = np.corrcoef(v1[:],v2[:])
#   return np.abs(corr[0,1])


# def calculte_DTWvsF(F_dist_mat,c_dist_mat):
#   """
#   Calculate DTW between coefficients versus the distance between sub-dynamics
#   """
#   np.fill_diagonal(F_dist_mat, np.nan)
#   np.fill_diagonal(c_dist_mat, np.nan)

#   return {'F_dist':F_dist_mat.flatten(),'c_dist': c_dist_mat.flatten()}
# def compute_DTW(vec1, vec2, options = {'window_size': 0.5}, method = 'sakoechiba' ):
#   """
#   Calculate DTW between 2 time signals
#   """
#   return dtw(vec1, vec2, method=method, options=options)

# def compute_distance_between_c(c_mat1, c_mat2 = [], normalize_c = True, func =spec_corr , pairs = [], type_return = 'mat'):
#   """
#   Compute distance between time coefficients (c)
#   """
#   if normalize_c:
#     c_mat1 = (c_mat1 -np.mean(c_mat1,1).reshape((-1,1)))/np.std(c_mat1,1).reshape((-1,1))
#     if len(c_mat2) > 0:
#       c_mat2 = (c_mat2 -np.mean(c_mat2,1).reshape((-1,1)))/np.std(c_mat2,1).reshape((-1,1))
#   if len(c_mat2) == 0: 
#     if len(pairs) == 0:
#       pairs = list(itertools.combinations(np.arange(np.shape(c_mat1)[0]),2))
#     dtw_dists = [func(c_mat1[pair[0],:],c_mat1[pair[1],:]) for pair in pairs]
#     if type_return == 'mat':
#       distance_mat = np.zeros((c_mat1.shape[0],c_mat1.shape[0]))
#     else:
#       distance_mat =[]
#     for count_dist, dist_u in enumerate(dtw_dists):
#       pair = pairs[count_dist]
#       if type_return == 'mat':
#         distance_mat[pair[0], pair[1]] = dist_u
#         distance_mat[pair[1], pair[0]] = dist_u
#       else:
#         distance_mat.append(dist_u)
#   else:
#     if len(pairs) == 0:
#       pairs = list(itertools.product(np.arange(np.shape(c_mat1)[0]),np.arange(np.shape(c_mat2)[0])))
#     dtw_dists = [func(c_mat1[pair[0],:],c_mat2[pair[1],:]) for pair in pairs]
#     if type_return == 'mat':
#       distance_mat = np.empty((c_mat1.shape[0],c_mat2.shape[0]))
#     else:
#       distance_mat = []
#     for count_dist, dist_u in enumerate(dtw_dists):
#       pair = pairs[count_dist]
#       if type_return == 'mat':
#         distance_mat[pair[0], pair[1]] = dist_u
#       else:
#         distance_mat.append(dist_u)
#   return distance_mat


  


# def plot_FvsC(F_dist_mat,c_dist_mat,ax = []):
#   """
#   Plot the distance between the sub-dynamics versus the distance between the corresponding cofficients. 
#   Inputs: 
#       F = list of sub-dynamics (each is a np.array of k X k)
#       c_mat = numpy array k X T 
#   """

#   c_dist_mat = np.array(c_dist_mat)
#   if isinstance(ax, list):
#     if len(ax) == 0:
#       fig, ax = plt.subplots()
#   if (F_dist_mat != F_dist_mat.T).any():
#     F_dist_mat = 0.5*np.nansum(np.dstack([F_dist_mat , F_dist_mat.T]),2)


#   if not (c_dist_mat == c_dist_mat.T).all():

#     c_dist_mat = 0.5*np.nansum(np.dstack([c_dist_mat , c_dist_mat.T]),2)

#   #c_dist_mat = 
#   np.fill_diagonal(c_dist_mat,np.nan)

#   np.fill_diagonal(F_dist_mat,np.nan)
#   dict_dist = calculte_DTWvsF(F_dist_mat,c_dist_mat)
#   nonan_map = np.isnan(dict_dist['F_dist'] ) == False
#   linear_model=np.polyfit(dict_dist['c_dist'][nonan_map], dict_dist['F_dist'][nonan_map],1)
#   linear_model_fn=np.poly1d(linear_model)

#   ax.plot(dict_dist['c_dist'][nonan_map],linear_model_fn(dict_dist['c_dist'][nonan_map]),color="green", alpha = 0.4, ls = '--')

#   ax.scatter(dict_dist['c_dist'][nonan_map], dict_dist['F_dist'][nonan_map], 100)

# def plot_fit(x,y):
#   """
#   Find a linear regression fit in the form of y = bx + c
#   """
#   linear_model=np.polyfit(x, y,1)
#   linear_model_fn=np.poly1d(linear_model)

#   return x, linear_model_fn(x)



# def create_dict_dist_c(coefficients, combs = [] , store_dict_c = {}, metrics = ['corr','DTW']):
#   """
#   Creates a dictionary with information regarding the distance between coefficients (c) of different sub-dynamics
#   """
#   if len(combs) == 0:    combs = list(itertools.combinations(np.arange(np.shape(coefficients)[0]),2))

#   if 'corr' in metrics:  store_dict_c['corr'] =  compute_distance_between_c(coefficients,pairs = combs, func = spec_corr,type_return = 'list')
#   if 'DTW' in metrics:  store_dict_c['DTW'] =  compute_distance_between_c(coefficients,pairs = combs, func = compute_DTW,type_return = 'list')
#   return store_dict_c
  

# def cross_corr_c(coefficients, combs = [] ):
#   """
#   Calculate the cross-correlation between coefficients
#   """
#   if len(combs) == 0:    combs = list(itertools.combinations(np.arange(np.shape(coefficients)[0]),2))
#   coeffs_corr = np.vstack([np.correlate(coefficients[pair[0],:], coefficients[pair[1],:], 'same').reshape((1,-1)) for pair in combs])
#   return coeffs_corr, combs


def update_D(former_D, step_D , x, y, reg1 = 0, reg_f= 0, bias_out_val = []) :
  """
  Update the matrix D by applying GD. Relevant just in case where D != I
  """
  if len(bias_out_val) == 0: bias_out_val = np.zeros((former_D.shape[0], 1))
  if reg1 == 0 and reg_f ==0:
    D = y @ linalg.pinv(x)
  else:
    basic_error = -2*(y - former_D @ x - bias_out_val) @ x.T
    if reg1 != 0:      reg1_error = np.sum(np.sign(former_D))
    else: reg1_error = 0      
    if reg_f != 0:      reg_f_error = 2*former_D
    reg_f_error = 0
    D = former_D - step_D *(basic_error + reg1*reg1_error + reg_f* reg_f_error)
  return D

def update_X(D, data, reg1 = 0, former_x = [], random_state = 0, other_params ={}):  
  """
  Update the latent dynamics. Relevant just in case where D != I
  """
  if reg1 == 0 :
    x = linalg.pinv(D) @ data
  else:
    clf = linear_model.Lasso(alpha=reg1,random_state=random_state, **other_params)
    clf.fit(D,y)
    x = np.array(clf.coef_)
  return x

def check_F_dist_init(F, max_corr = 0.1):
    """
    This function aims to validate that the matrices in F are far enough from each other
    """
    combs = list(itertools.combinations(np.arange(len(F)),2))
    corr_bool = [spec_corr(F[comb_s[0]],F[comb_s[1]]) > max_corr for comb_s in combs]
    counter= 100
    while (corr_bool == False).any():
        counter +=1
        for comb_num,comb in enumerate(combinations):
            if spec_corr(F[comb[0]],F[comb[1]])  > max_corr:
                fi_new = init_mat(np.shape(F[0]),dist_type = 'norm',r_seed = counter)
                F[comb[0]] = fi_new
    return F
        
    
    
#%% Main Model Training
def train_model_include_D(max_time = 500, dt = 0.1, dynamics_type = 'cyl',num_subdyns = 3, 
                          error_reco = np.inf,error_step_max  = 15, error_order = np.nan, data = [], same_c = True,step_f = 30, 
                          GD_decay = 0.85, weights_orders = [],clean_dyn = [],max_error = 1e-3,grad_vec = [], 
                          max_iter = 3000, F = [], coefficients = [], params= {'update_c_type':'inv','reg_term':0,'smooth_term':0}, 
                          epsilon_error_change = 10**(-5), D = [], 
                          x_former =[], latent_dim = None, include_D  = False,step_D = 30, reg1=0,reg_f =0 , 
                          max_data_reco = 1e-3, acumulated_error = False, sigma_mix_f = 0.1, error_step_add = 120, 
                          action_along_time = 'median', error_step_max_display = 8, to_print = True, seed = 0, seed_f = 0, 
                          return_evolution = False,  normalize_eig  = True,
                          params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}, start_sparse_c = False,
                          init_distant_F = False,max_corr = 0.1, decaying_reg = 1, center_dynamics = False, bias_term = False, bias_out = False,
                          other_params_c={}, include_last_up = False):
    
  """
  This is the main function to train the model! 
  Inputs:
      max_time      = Number of time points for the dynamics. Relevant only if data is empty;
      dt            =  time interval for the dynamics
      dynamics_type = type of the dynamics. Can be 'cyl', 'lorenz', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      num_subdyns   = number of sub-dynamics
      error_reco    = intial error for the reconstruction (do not touch)
      error_step_max= the step of the weights given to errors from different orders
      error_order   = error of the order
      data          = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
      same_c        = if there is more than one sample for the dynamics (for instance - noisy case), than whether to find a shared coefficients representation to all samples (irrelevant if only one sample) 
      step_f        = initial step size for GD on the sub-dynamics
      GD_decay      = Gradient descent decay rate
      weights_orders= only use if you have a pre-defined set of weights for the different orders
      clean_dyn     = use if the dynamics in data is not clean (e.g. noisy scenario). Otherwise - keep empty.
      max_error     = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
      grad_vec      = the amount by which the curve in 'weights_orders' will change towards higher orders
      max_iter      = # of max. iterations for training the model
      F             = pre-defined sub-dynamics. Keep empty if random.
      coefficients  = pre-defined coefficients. Keep empty if random.
      params        = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
      epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
      D             = pre-defined D matrix (keep empty if D = I)
      x_former      = IGNORE; NEED TO ERASE! (NM&&&&)
      latent_dim    =  If D != I, it is the pre-defined latent dynamics.
      include_D     = If True -> D !=I; If False -> D = I
      step_D        = GD step for updating D, only if include_D is true
      reg1          = if include_D is true -> L1 regularization on D
      reg_f         = if include_D is true ->  Frobenius norm regularization on D
      max_data_reco = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
      acumulated_error       = whether to check a k_th order error or the acumulated error (True = accumulated, False = ordered error)
      sigma_mix_f            = std of noise added to mix f
      error_step_add         = consider a new order only after passing error_step_add  iterations. Do not touch. 
      action_along_time      = the function to take on the error over time. Can be 'median' or 'mean'
      error_step_max_display = error order to print when training (int > 0)
      to_print               = to print error value while training? (boolean)
      seed                   = random seed
      seed_f                 = random seed for initializing f
      return_evolution       = store the evolution of the training (does not change the model, but can be very heavy so recommneded False unless the evolution is needed)
      normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
      params_ex              = parameters related to the creation of the ground truth dynamics. e.g. {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
      start_sparse_c         = If true - start with sparse c and then infer F. If False - start with random F and infer c (not necessarily sparse)
      init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
      max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
      decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
      center_dynamics        = whether to shift the dynamics to be centered around (0,0). (boolean)                                                                                                                       
      bias_term              = whether to add a bias term to the model, in the form of x_(t+1) = \sum(f_i c_i)* x_t + bias (boolean)
      bias_out               = in cases where D!=I, y = Dx + bias_out
  """  
  if acumulated_error  and include_D: raise ValueError('When including D, the error should not be cumulative (you should set the acumulated_error input to False')
  
  if error_step_max > 1 and include_D: 
    print('Error step was reduced to 1 since D is updated')
    error_step_max = 1
    error_step_max_display = 1
  if len(weights_orders) == 0: weights_orders = np.linspace(1,2**error_step_max,error_step_max)[::-1]
  if len(grad_vec) == 0: grad_vec = np.linspace(0.99,1.01,error_step_max)
  if not include_D and len(data) > 1: latent_dyn = data
  if include_D and bias_term and bias_out:
      print('Disabling internal bias term since D ~=I and bias_out is true')
      bias_term = False
  if not include_D and bias_out: # disable bias out if D = I
      bias_out = False
      

  if return_evolution:
      store_iter_restuls = {'coefficients':[], 'F':[],'L1':[]}
  step_f_original = step_f
  
  # Define data and number of dyns
  if len(data) == 0 :
    data            = create_dynamics(type_dyn = dynamics_type, max_time = max_time, dt = dt, params_ex = params_ex)
    if not include_D: latent_dyn = data
    one_dyn = True
  else:
    if isinstance(data, np.ndarray): 
      if not include_D: latent_dyn = data
      one_dyn = True
    else:       
      if len(data) == 1: 
        one_dyn = True
        data = data[0]
        if not include_D: latent_dyn          =data #[0]
      else: 
        one_dyn = False

  if include_D:
    if isinstance(data, list):
      latent_dim = int(np.max([data[0].shape[0] / 5,3])); n_times = data[0].shape[1]
    else:
      latent_dim = int(np.max([data.shape[0] / 5,3]));n_times = data.shape[1]
  else:
    if isinstance(data, list):
      latent_dim = data[0].shape[0] ; n_times = data[0].shape[1]
    else:
      latent_dim = data.shape[0]; n_times = data.shape[1]

  if include_D: # Namely - model need to study D
    if one_dyn:
      if len(D) == 0: D = init_mat(size_mat = (data.shape[0], latent_dim) , dist_type ='sparse', init_params={'k':4})
      elif D.shape[0] != data.shape[0]: raise ValueError('# of rows in D should be = # rows in the data ')
    else:
      D = init_mat(size_mat = (data[0].shape[0], latent_dim) , dist_type ='sparse')
      D = [D]*len(data)
  else:
    latent_dyns = data

  if len(F) == 0:            
      F              = [init_mat((latent_dim, latent_dim),normalize=True,r_seed = seed_f+i) for i in range(num_subdyns)]
      if init_distant_F:
           F = check_F_dist_init(F, max_corr = max_corr)
 
  """
  Initialize Coeffs
  """
  if len(coefficients) == 0: 
      if one_dyn or same_c:
          if start_sparse_c:
              coefficients   = init_mat((num_subdyns,n_times-1),dist_type = 'regional')
          else:
              coefficients   = init_mat((num_subdyns,n_times-1))
      else:
          coefficients   = init_mat((num_subdyns,n_times-1,len(latent_dyns)))

  if len(params) == 0:       params         = {'update_c_type':'inv','reg_term':0,'smooth_term':0}
  counter               = 1
  
  error_reco_all            = np.inf*np.ones((1,error_step_max))
  error_reco_all_med        = np.inf*np.ones((1,error_step_max))
  if not include_D:
    if one_dyn:
      cur_reco              = create_reco(latent_dyn=latent_dyn, coefficients= coefficients, F=F, accumulation = acumulated_error)
    else:
      if same_c:
        cur_reco              = np.dstack([create_reco(latent_dyn=latent_dyn_i, coefficients= coefficients, F=F, accumulation = acumulated_error) for latent_dyn_i in latent_dyns])
      else:
        cur_reco              = np.dstack([create_reco(latent_dyn=latent_dyn_i, coefficients= coefficients[:,:,samp_num], F=F, accumulation = acumulated_error) for samp_num, latent_dyn_i in enumerate(latent_dyns)])
 
  error_reco_array      = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
  error_reco_array_med  = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
  if np.isnan(error_order): 
    error_order           = error_step_max                                                                                     # if < error_step_max -> do not consider all orders initially
  counter_change_order  = 1
  if include_D:
    data_reco_error  = np.inf
  else:
    data_reco_error = -np.inf
    

  """
  Center dynamics
  """
  if center_dynamics:
      if one_dyn:  to_center_vals = -np.mean(data)
      else:        to_center_vals = [-np.mean(dyn) for dyn in data]
      if include_D:
          if one_dyn:  data = data + to_center_vals
          else:        
              data = [data_spec + to_center_vals[i] for i, data_spec in enumerate(data)]
      else:          
          if one_dyn:  latent_dyn = latent_dyn + to_center_vals
          else:        
              latent_dyns = [latent_dyn + to_center_vals[i] for i, latent_dyn in enumerate(latent_dyns)]
      
  else:
    to_center_vals = 0      
  while ((error_reco_all_med[0,error_step_max-1] > max_error) or (data_reco_error > max_data_reco)) and (counter < max_iter):
      
    ### Store Iteration Results
    if return_evolution:
        store_iter_restuls['F'].append(F)
        store_iter_restuls['coefficients'].append(coefficients)
        store_iter_restuls['L1'].append(np.sum(np.abs(coefficients),1))
    if (counter_change_order == error_step_add) and (error_order < error_step_max):
      error_order +=1
      counter_change_order = 1
      step_f = step_f_original*(GD_decay**(error_order**2))      
    else:
      counter_change_order += 1
      
    """
    Update x
    """
    
    if include_D:
      if one_dyn: latent_dyn = update_X(D, data,random_state=seed)
      else: latent_dyns = [update_X(D, data_i,random_state=seed) for  data_i in data]
      
    """
    Decay reg 
    """
    if params['update_c_type'] == 'lasso':
        params['reg_term'] = params['reg_term']*decaying_reg 
    
    """
    Update coefficients
    """
    if counter != 1 or not start_sparse_c:
        if one_dyn:
          if acumulated_error:
            coefficients = update_c(F, cur_reco, params, clear_dyn= latent_dyn, direction = 'n2c',random_state=seed,other_params=other_params_c )
          else:
            coefficients = update_c(F,latent_dyn, params,random_state=seed,other_params=other_params_c)
        else:
          if same_c:
            if acumulated_error:
              coefficients = np.mean(np.dstack([update_c(F, cur_reco[:,:,i], params, clear_dyn= latent_dyn, direction = 'n2c',other_params=other_params_c) for i in range(cur_reco.shape[2])]),2)
            else:
              coefficients = np.mean(np.dstack([update_c(F,latent_dyns[i], params,other_params=other_params_c) for i in range(cur_reco.shape[2])]),2)
          else:
            if acumulated_error:
              coefficients = np.dstack([update_c(F, cur_reco[:,:,i], params, clear_dyn= latent_dyn, direction = 'n2c',other_params=other_params_c) for i in range(cur_reco.shape[2])])
            else:
              coefficients = np.dstack([update_c(F,latent_dyns[i], params,other_params=other_params_c) for i in range(cur_reco.shape[2])])
        
    """
    Update bias_out
    """
    if bias_out:
        if one_dyn:
            bias_out_val = update_bias_out(latent_dyn, data ,D,action_along_time= action_along_time)
        else:
            bias_out_val = [update_bias_out(latent_dyns[i],data_spec, D,action_along_time= action_along_time) for i,data_spec in enumerate(data)]
    else:
        if one_dyn:
            bias_out_val = np.zeros((data.shape[0], 1))
        else:
            bias_out_val = [np.zeros((data_spec.shape[0], 1)) for i,data_spec in enumerate(data)]
    
    """
    Update D
    """
    
    if include_D:
      if one_dyn: D = update_D(D, step_D, latent_dyn, data, reg1,reg_f, bias_out_val) 
      else: D = [update_D(D_i, step_D, latent_dyns[i], data[i], reg1,reg_f, bias_out_val) for i, D_i in enumerate(D)]
    
    
    """
    Update bias_term
    """
    if bias_term:
        if one_dyn:
            bias_val = update_bias(latent_dyn, F,coefficients,action_along_time= action_along_time)
        else:
            if same_c:
                bias_val = np.dstack([update_bias(latent_dyn, F,coefficients,action_along_time= action_along_time) for i, latent_dyn in enumerate(latent_dyns)]).mean(2)
            else:
                bias_val = [update_bias(latent_dyn, F,coefficients[:,:,i],action_along_time= action_along_time) for i,latent_dyn in enumerate(latent_dyns)]
    else:
        if one_dyn:
            bias_val = np.zeros((latent_dyn.shape[0], 1))
        else:
            if same_c:                
                bias_val = np.zeros((latent_dyns[0].shape[0], 1))
            else:
                bias_val = [np.zeros((latent_dyn.shape[0], 1)) for i,latent_dyn in enumerate(latent_dyns)]


        
    """
    Update F
    """
    
    if one_dyn:
      F = update_f_all(latent_dyn,F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, normalize_eig = normalize_eig , bias_val = bias_val )

    else:
      if same_c:
        F_lists = [update_f_all(latent_dyns[i],F,coefficients,step_f,normalize=False, acumulated_error = acumulated_error, error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, bias_val = bias_val) for i in range(len(latent_dyns))]
        store_F = np.zeros((latent_dyns[0].shape[0], latent_dyns[0].shape[0],num_subdyns))
        for F_list in F_lists:
          store_F = store_F + np.dstack(F_list)
        store_F = store_F / len(F_lists)
        F = list(store_F.T)
      else:
        F_lists = [update_f_all(latent_dyns[i],F,coefficients[:,:,i],step_f,normalize=False, acumulated_error = acumulated_error, error_order = error_order-1, action_along_time= action_along_time, weights = weights_orders, bias_val = bias_val[i]) for i in range(len(latent_dyns))]
        store_F = np.zeros((latent_dyns[0].shape[0], latent_dyns[0].shape[0],num_subdyns))
        for F_list in F_lists:
          store_F = store_F + np.dstack(F_list)
        store_F = store_F / len(F_lists)
        F = list(store_F.T)        

    weights_orders = weights_orders * grad_vec
    step_f *= GD_decay
    if one_dyn or len(clean_dyn) == 0:
      mid_reco = latent_dyn
    else:
      mid_reco = clean_dyn
      
    #mid_reco
    error_reco_all = np.inf*np.ones((1,max(error_step_max_display,error_step_max))) #[]
    error_reco_all_med = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
    if include_D:
      if one_dyn:       data_reco_error = np.mean((data - D @ latent_dyn)**2)
      else:  data_reco_error = np.mean((data - np.mean(np.dstack([D @ latent_dyn for latent_dyn in latent_dyns]),2))**2)
    for n_error_order in range(max(error_step_max_display,error_step_max)):     
        try:
            mid_reco = create_reco(mid_reco, coefficients, F, acumulated_error)
            error_reco = np.mean((latent_dyn -mid_reco)**2)
            error_reco_all[0,n_error_order] = error_reco
            error_reco_all_med[0,n_error_order] = np.median((latent_dyn -mid_reco)**2)
        except:
            print('mid_reco does not work')

    error_reco_array = np.vstack([error_reco_array,np.array(error_reco_all).reshape((1,-1))])
    error_reco_array_med = np.vstack([error_reco_array_med,np.array(error_reco_all_med).reshape((1,-1))])
    if np.mean(np.abs(np.diff(error_reco_array[-5:,:],axis = 0))) < epsilon_error_change:
      F = [f_i + sigma_mix_f*np.random.randn(f_i.shape[0],f_i.shape[1]) for f_i in F]
      print('mixed F')

    if to_print:
        print('Error:    ' + '; '.join(['order' + str(i) + '=' + str(error_reco_all[0,i]) for i in range(   min(len(error_reco_all[0]),error_step_max_display))]))
        print('Error med:' + '; '.join(['order' + str(i) + '=' + str(error_reco_all_med[0,i]) for i in range(min(len(error_reco_all_med[0]),error_step_max_display) )]))
        if include_D:
            print('Error recy y:' + '; '.join(['order' + str(j) + '=' + str(data_reco_error) for j in range(1)]))

    counter += 1

  if counter == max_iter: print('Arrived to max iter')
  #params['to_norm_fx'] = False
  if include_last_up:
      coefficients = update_c(F, latent_dyn,params,  {'reg_term': 0, 'update_c_type':'inv','smooth_term' :0, 'num_iters': 10, 'threshkind':'soft'})
  else:
      coefficients = update_c(F, latent_dyn, params,other_params=other_params_c)  
  if return_evolution:
    store_iter_restuls['F'].append(F)
    store_iter_restuls['coefficients'].append(coefficients)
    store_iter_restuls['L1'].append(np.sum(np.abs(coefficients),1))
  
  
  if center_dynamics:
      if include_D:
          if one_dyn:  bias_out_val = bias_out_val - to_center_vals
          else:        
              bias_out_val = [bias_out_val - to_center_vals[i] for i in range(len(to_center_vals))]
      else:          
          if one_dyn:  bias_val = bias_val + to_center_vals
          else:   
              if same_c:
                  bias_val = [bias_val + to_center_vals[i] for i in range(len(to_center_vals))]
              else:
                  bias_val = [bias_val[i] + to_center_vals[i] for i in range(len(to_center_vals))]
          
  additional_return = {'bias_val': bias_val, 'to_center_vals': to_center_vals,'bias_out_val':bias_out_val}          
  if not return_evolution:
      if not include_D: D = [];
      if one_dyn:      return coefficients, F, latent_dyn, error_reco_array, error_reco_array_med,D,additional_return
      else:  return coefficients, F, latent_dyns, error_reco_array, error_reco_array_med,D,additional_return
  else:
      if not include_D: D = [];
      if one_dyn:      return coefficients, F, latent_dyn, error_reco_array, error_reco_array_med,D,store_iter_restuls, additional_return
      else:  return coefficients, F, latent_dyns, error_reco_array, error_reco_array_med,D,store_iter_restuls,additional_return
      







#%% Saving
  
def check_save_name(save_name, invalid_signs = '!@#$%^&*.,:;', addi_path = [], sep=sep)  :
    """
    Check if the name is valid
    """
    for invalid_sign in invalid_signs:   save_name = save_name.replace(invalid_sign,'_')
    if len(addi_path) == 0:    return save_name
    else:   
        path_name = sep.join(addi_path)
        return path_name +sep +  save_name

def save_file_dynamics(save_name, folders_names,to_save =[], invalid_signs = '!@#$%^&*.,:;', sep  = sep , type_save = '.npy'):
    """
    Save dynamics & model results
    """                  
    save_name = check_save_name(save_name, invalid_signs)
    path_name = sep.join(folders_names)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if type_save == '.npy':
        if not save_name.endswith('.npy'): save_name = save_name + '.npy'
        np.save(path_name +sep +  save_name, to_save)
    elif type_save == '.pkl':
        if not save_name.endswith('.pkl'): save_name = save_name + '.pkl'
        dill.dump_session(path_name +sep +  save_name)
    # elif type_save == '.pkl_save':
    #     if not save_name.endswith('.pkl'): save_name = save_name + '.pkl'
    #     filehandler = open(b"Fruits.obj","wb")
    #     pickle.dump(banana,filehandler)


import pickle
def saveLoad(opt,filename):
    global calc
    if opt == "save":
        f = open(filename, 'wb')
        pickle.dump(calc, f, 2)
        f.close
     
    elif opt == "load":
        f = open(filename, 'rb')
        calc = pickle.load(f)
    else:
        print('Invalid saveLoad option')
        
def load_vars(folders_names ,  save_name ,sep=sep , ending = '.pkl',full_name = False):
    """
    Load results previously saved
    Example:
        load_vars('' ,  'save_c.pkl' ,sep=sep , ending = '.pkl',full_name = False)
    """
    if full_name: 
        dill.load_session(save_name)    
    else:
        if len(folders_names) > 0: path_name = sep.join(folders_names)
        else: path_name = ''
      
        if not save_name.endswith(ending): save_name = '%s%s'%(save_name , ending)
        dill.load_session(path_name +sep +save_name)

    
        

 
def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')
    return str_to_change


#%% Plots
def compare_orders(latent_dyn, coefficients, F, max_delay_plot, max_to_run, to_plot = True, axs = [], params_plot = {},interval_show = 0):
  """
  Compare the reconstruction of the dynamics under different reconstruction orders
  """
  if interval_show == 0 : interval_show = int(max_delay_plot/10)
  max_to_run = np.max([max_delay_plot, max_to_run])
  delays_options = np.arange(0,max_delay_plot,interval_show)
  if isinstance(axs,list):
    if len(axs) == 0:
        if latent_dyn.shape[0] == 3:
            fig, axs = plt.subplots(2,int(np.ceil(len(delays_options)/2)),figsize = (35,15), subplot_kw={'projection':'3d'})  #, sharex  = True, sharey = True
        elif latent_dyn.shape[0] == 2:
            fig, axs = plt.subplots(2,int(np.ceil(len(delays_options)/2)),figsize = (35,15))
        else:
            raise ValueError('Invalid dimension for the dynamics')
        
  axs_flat = axs.flatten()
  mid_reco = latent_dyn
  counter_plot =0
  error_orders = [0]
  for level_reco in range(max_to_run+1):
    mse = (np.mean((latent_dyn - mid_reco)**2))**0.5
    if to_plot:
      if level_reco in delays_options:    
        visualize_dyn(mid_reco,axs_flat[counter_plot],params_plot); axs_flat[counter_plot].set_title('Reconstructed order %g, rmse:%g'%(level_reco, mse))
        counter_plot += 1    
    mid_reco = create_reco(mid_reco,coefficients, F)
    error_orders.append(mse)
  plt.subplots_adjust(hspace = 0.2, wspace=0.2)
  fig,ax = plt.subplots()
  ax.plot(error_orders,'*-'); add_labels(ax, xlabel ='Reconstruction Order', ylabel = 'rMSE',zlabel = None); ax.set_yscale('log'); ax.axhline(1,color = 'r', ls = '--',alpha = 0.3)
  plt.suptitle('Reconstruction under different reconstruction orders',fontsize = 16)  
  plt.subplots_adjust()
  return error_orders, mid_reco, level_reco    

def find_closest(vec1, vec2, metric = 'mse'):
    """
    find to which elements in vec2 each element in vec1 is the closest
    """
    if metric == 'mse':
        tiled_vec1 = np.tile(vec1.reshape((1,-1)), [len(vec2),1]) 
        tiled_vec2 = np.tile(vec2.reshape((1,-1)), [len(vec1),1]).T
        v1_closest_to_v2_args = np.argmin((tiled_vec1 - tiled_vec2)**2, 1)
        v1_closest_to_v2 = vec1[v1_closest_to_v2_args]
        return v1_closest_to_v2, v1_closest_to_v2_args


#%% Compare Initial Conditions
def cal_f_lst(F_init_list, ind = -1):
    """
    Extract the final results for F in cases were the training returned the evolution results as well
    """
    F_lasts = [f_list_i[ind] for f_list_i in F_init_list]
    return F_lasts

def plot_f_different_initialization(F_init_list, ax = [], ind = -1,annot = True):
    """
    Plot the set of sub-dynamics that were obtained under different initializations
    """
    if isinstance(F_init_list[0][0],list):
        F_lasts = cal_f_lst(F_init_list, ind = ind)
    else:
        F_lasts = F_init_list
    if isinstance(ax,list):
        if len(ax) == 0:
            fig, ax = plt.subplots(len(F_init_list),len(F_lasts[0]), sharex = True,sharey = True,figsize = (len(F_lasts[0])*5,len(F_init_list)*5))        
    [plot_subs(F_last,ax[i,:],annot = annot) for i,F_last in enumerate(F_lasts)]
    return F_lasts

def match_corrs(F1,F2,c2 = []):
    """
    Match different pairs of sub-dynamics based on correlation. Organize F2 by F1
    Inputs:
        F1   = list of np.arrays, each np.array is kXk
        F2   = list of np.arrays, each np.array is kXk
        c2   = the coefficients associated with the sub-dynamics F2
    Outputs:
        F1  = same as input
        F2_org = ordered list of the F2 sub-dynamics, ordered according to the correlation with F1
        c2 = ordered c2, ordered according to the correlation with F1
    """
    store_corr = np.zeros((len(F1),len(F2)))
    store_best = {} # keys are inds of F1, vals of F2
    for i1, f1_i in enumerate(F1):
        for i2, f2_i in enumerate(F2):
            corr_cur = spec_corr(f1_i.flatten(), f2_i.flatten())
            store_corr[i1,i2] = corr_cur
    while np.sum(store_corr) > 0:
        B = np.unravel_index(np.argmax(store_corr, axis=None), store_corr.shape)
        #store_best.append(B)
        store_best[B[0]]   = B[1]
        store_corr[B[0],:] = 0
        store_corr[:,B[1]] = 0

    order_keys = np.sort(np.array(list(store_best.keys())))
    F2_org = [F2[store_best[key]] for key in order_keys]
    if len(c2) > 0:    c2 = [c2[store_best[key],:] for key in order_keys]
    return F1, F2_org, c2
        
        
        
def check_initialization(F_init_list,coeffs_init_list,error_reco_init_list,num_subdyns, ax = [], ax_eigen = [], ax_co = [], ax_co_heat = [], error_max_show = 15,init_point = 70,ax_reco =[],reco_order = 10, latent_dyn_init_list = [], annot = True,name_var = 'IC'):
    """
    Check, explore and visualize changes in initial conditions effects
    
    """
    num_subdyns = len(F_init_list[0])
    if isinstance(ax,list):
        if len(ax) == 0:
          fig, ax = plt.subplots(1,num_subdyns, figsize = (num_subdyns*8,int(0.4*len(F_init_list))))
    if isinstance(ax_co,list):
        if len(ax_co) == 0:
          fig_co, ax_co = plt.subplots(1,num_subdyns, figsize = (num_subdyns*5,int(0.3*len(F_init_list))))
    if isinstance(ax_co_heat,list):
        if len(ax_co_heat) == 0:
          fig_co_heat, ax_co_heat = plt.subplots(1,num_subdyns, figsize = (num_subdyns*8,int(0.3*len(F_init_list))))

    F_lasts = plot_f_different_initialization(F_init_list,annot = annot)
    if len(latent_dyn_init_list) > 0:
        if isinstance(ax_reco,list):
            if len(ax_reco) == 0:
                
              fig_reco, ax_reco = plt.subplots(2,len(F_lasts), figsize = (num_subdyns*8,10) , subplot_kw={'projection':'3d'})
    if isinstance(F_init_list[0][0],list):
        c_lasts = cal_f_lst(coeffs_init_list)
    else: 
        c_lasts = coeffs_init_list

    Fpair1 = F_lasts[0]
    F_lasts_orgs = [Fpair1]
    c1 = c_lasts[0]
    c_lasts_orgs = [c1]
    for idf in range(len(F_lasts)-1):        
        Fpair2 = F_lasts[idf+1]
        c2    = c_lasts[idf+1]
        _,Fpair1,c2org = match_corrs(Fpair1,Fpair2, c2) 
        F_lasts_orgs.append(Fpair1)
        c_lasts_orgs.append(c2org)
    all_stacked_subdyns = []    

    for sub_dyn_num in range(num_subdyns):
        stacked_subdyns = np.vstack([F_lasts_orgs[i][sub_dyn_num].flatten() for i in range(len(F_lasts_orgs))])
        sns.heatmap(np.abs(np.corrcoef(stacked_subdyns)),annot = True,ax = ax[sub_dyn_num],cmap = 'Greens')
        all_stacked_subdyns.append(stacked_subdyns)
    fig.suptitle('Subdynamics correlations for different %s'%name_var)
    [add_labels(ax = ax[i], title = 'f%g'%i, xlabel = '%s Iteration #'%name_var, ylabel = '%s Iteration #'%name_var, zlabel = None) for i in range(num_subdyns)]

    colors = np.random.rand(3,num_subdyns)
    [check_eigenspaces(F_lasts_orgs[init_num], colors = colors, ax = [], title2 = 'Eigenspaces (%s#%g)'%(name_var,init_num),title1= 'Eigenvalues (%s#%g)'%(name_var,init_num)) for init_num in range(len(F_init_list))];

    [[ax_co[sub_dyn_spec].plot(c_lasts_org[sub_dyn_spec][init_point:], alpha = 0.4) for sub_dyn_spec in range(num_subdyns)] for c_lasts_org in c_lasts_orgs]
    ax_co[-1].legend(['%s#%g'%(name_var,num_IC) for num_IC in range(len(c_lasts_orgs))])
    [add_labels(ax_spec, title = 'c#%g'%sub_dyn_num, xlabel = 'Time',ylabel = 'Coeffs',zlabel = None) for sub_dyn_num, ax_spec in enumerate(ax_co)]
    fig_co.suptitle('Coefficients for different %s'%name_var)
    

    [sns.heatmap(np.vstack([c_lasts_org[sub_dyn_spec][init_point:]for c_lasts_org  in c_lasts_orgs]) , ax = ax_co_heat[sub_dyn_spec], alpha = 0.4)  for sub_dyn_spec in range(num_subdyns) ]
    [add_labels(ax_spec, title = 'c#%g'%sub_dyn_num, xlabel = 'Time',ylabel = 'Different IC',zlabel = None) for sub_dyn_num, ax_spec in enumerate(ax_co_heat)]
    fig_co.suptitle('Coefficients for different %s, each subplot i describes the coefficients corresponding to sub-dynamic i'%name_var)

    fig, ax_bar = plt.subplots(figsize = (7,7))
    error_max_show = np.min([error_max_show, error_reco_init_list[0].shape[1]])
    last_error = np.vstack([error_reco_init_list[num_IC][-1,:error_max_show] for num_IC in range(len(F_init_list))])
    columns=['order %g'%order for order in range(error_max_show)]
    pd.DataFrame(last_error,columns = columns, index = ['%s #%g'%(name_var,iteration) for iteration in range(len(F_init_list))]).T.plot(ax = ax_bar)
    ax_bar.set_yscale('log')
    add_labels(ax_bar, xlabel = 'Reconstruction Order', ylabel = 'Error',zlabel = None, title = 'Error for different reconstruction orders, under different initial conditions' )
    #    
    if len(latent_dyn_init_list) > 0:
        
        [visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts[i],F_lasts[i], step_n = 1), ax = ax_reco[0,i], color_by_dominant = True, coefficients =c_lasts[i]) for i in range(len(F_lasts)) ]

        [visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts[i],F_lasts[i], step_n = reco_order), ax = ax_reco[1,i], color_by_dominant = True, coefficients =c_lasts[i]) for i in range(len(F_lasts)) ]


        [add_labels(ax = ax_reco[0,i], title = 'reco order 1, sample #%g'%i) for i in range(len(F_lasts)) ]
        #[visualize_dyn(create_reco(latent_dyn_init_list[i], c_lasts_orgs[i],F_lasts_orgs[i], step_n = reco_order), ax = ax_reco[1,i], color_by_dominant = False, coefficients =c_lasts_orgs[i]) for i in range(len(F_lasts_orgs)) ]
        [add_labels(ax = ax_reco[1,i], title = 'reco order %g, sample #%g'%(reco_order, i)) for i in range(len(F_lasts)) ]
        
    return F_lasts_orgs,c_lasts_orgs,all_stacked_subdyns
        
def create_colors(len_colors, perm = [0,1,2]):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    colors = np.vstack([np.linspace(0,1,len_colors),(1-np.linspace(0,1,len_colors))**2,1-np.linspace(0,1,len_colors)])
    colors = colors[perm, :]
    return colors
        
def plot_dict_array(dict_to_plot, cmap = 'PiYG', axs = [], key_to_plot = 'coefficients',type_plot = 'plot',min_time = 50,sharey= 'row',rows_plot = -10,logscale = False, zero_ref = 0,xlabel = 'Time',ylabel = 'coeffs'):
    """
    Plot dynamics with different regularization values
    type_plot: can be plot or heatmap
    """
    
    if isinstance(axs,list):
        if len(axs) == 0:
            fig, axs = plt.subplots(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]), figsize = (15,len(dict_to_plot.keys())*4), sharey = sharey)  #, sharex  = True, sharey = True
            axs = axs.reshape(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]))
    reg_ordered_to_plot = list(dict_to_plot.keys())
    num_dyns_values = dict_to_plot[reg_ordered_to_plot[0]]
    if type_plot == 'plot':
       
        if rows_plot <= -5:         [[axs[reg_val_num,num_dyns_count].plot(dict_to_plot[reg_val][num_dyns_val][key_to_plot][:,min_time:].T) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        else:         
            [[axs[reg_val_num,num_dyns_count].plot(dict_to_plot[reg_val][num_dyns_val][key_to_plot][rows_plot, min_time:]) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        if not np.isnan(zero_ref):
            [ax.axhline(zero_ref,color = 'r',ls = '--',alpha = 0.5) for ax in axs.flatten()]

            
    elif    type_plot == 'heat':
        if rows_plot <= -5: [[sns.heatmap(dict_to_plot[reg_val][num_dyns_val][key_to_plot][:,min_time:].T,ax = axs[reg_val_num,num_dyns_count],vmin = 0,vmax = 0.1,cmap = cmap) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
        else:  [[sns.heatmap(dict_to_plot[reg_val][num_dyns_val][key_to_plot][rows_plot,min_time:].reshape((1,-1)),ax = axs[reg_val_num,num_dyns_count],vmin =0,vmax = 0.1, cmap = cmap) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
       
    else:
        raise NameError('Unknown type plot!')
    [[add_labels(ax = axs[reg_val_num,num_dyns_count], title = 'reg =%g, for %g dynamics'%(reg_val, num_dyns_val), xlabel = xlabel,ylabel = ylabel,zlabel = None) for num_dyns_count, num_dyns_val in enumerate(num_dyns_values.keys())] for reg_val_num, reg_val in enumerate(reg_ordered_to_plot)]
    if logscale:
        [ax_spec.set_yscale('log') for ax_spec in axs.flatten()]
    fig.subplots_adjust(hspace = 0.7,wspace = 0.5)    
    
    
def plotfig(dict_to_plot,axs = [],cmap_base = 'viridis',name1 = ' reg',name2 = ' # sub-dyns',step_n = 1, accumulation = False,params_plot = {} , suptitle = ''):
    """
    Plot specific dynamics
    dict_to_plot: a dictionary with the dynamics to plot
    """
    if isinstance(axs,list):
        if len(axs) == 0:
            fig, axs = plt.subplots(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]), figsize = (15,len(dict_to_plot.keys())*5) , subplot_kw={'projection':'3d'})  #, sharex  = True, sharey = True
            axs = axs.reshape(len(dict_to_plot.keys()), len(dict_to_plot[list(dict_to_plot.keys())[0]]))

    cmap = plt.cm.get_cmap(cmap_base, 3)
    [[visualize_spec_dyn(dict_to_plot[reg_val][sub_dyn_val],step_n = step_n, ax = axs[reg_num,sub_dyn_num],  accumulation = accumulation, cmap = cmap,params_plot = {**{'title':'%s%g%s%g'%(name1, reg_val, name2,sub_dyn_val)},**params_plot}) for sub_dyn_num, sub_dyn_val in enumerate(value_full_sub.keys())] for reg_num, (reg_val,value_full_sub) in enumerate(dict_to_plot.items())]
    if len(suptitle)>0: fig.suptitle(suptitle)

def visualize_spec_dyn(part_dic,ax, step_n = 1, accumulation = False,cmap = 'PiYG',return_fig  = True,params_plot = {}):
    """
    Visualize a set of reconstructed dynamics
    Inputs:
        part_dic = dictionary with keys  'latent_dyn', 'coefficients', 'F'
        ax        = subplot to plot into
        step_n   = order of the reconstruction
        accumulation = whether the reconstruction should be limited by order or a full reconstruction
        
    """
    visualize_dyn(create_reco(part_dic['latent_dyn'],part_dic['coefficients'],part_dic['F'],step_n = step_n,accumulation = accumulation)[:,:-1], turn_off_back=True, color_by_dominant = True, coefficients = part_dic['coefficients'], ax = ax,cmap = cmap, return_fig = return_fig, colorbar = False, params_plot = params_plot)    
    
    
def calcul_contribution(reco, real, direction = 'forward',dict_store ={}, func = np.nanmedian):
    """
    Calculate the error and the % close points for specific reconstruction matrix
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        direction: can be forward or backward
        func: the function to apply on the relative error of each point
    Outputs:
        error: relative error
        percent_close: % of points which are within the range
    """
    error = relative_eror(reco,real, return_mean = True, func = func)
    if direction == 'forward': error = 1-error
    elif direction == 'backward': error = error
    else: raise NameError('Unknown direction')
    percent_close = claculate_percent_close(reco, real)
    if direction == 'forward': percent_close =percent_close
    elif direction == 'backward': percent_close =  1-percent_close
    else: raise NameError('Unknown direction')

    return error,percent_close

def relative_eror(reco,real, return_mean = True, func = np.nanmean):
    """
    Calculate the relative reconstruction error
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        return_mean: reaturn the average of the reconstruction error over time
        func: the function to apply on the relative error of each point
    Output:
        the relative error (or the mean relative error over time if return_mean)
    """
    error_point = np.sqrt(((reco - real)**2)/(real)**2)
    if return_mean:
        return func(error_point )
    return func(error_point,0)


def claculate_percent_close(reco, real, epsilon_close = 3, return_quantiles = False, quantiles = [0.05,0.95]):
    """
    Calculte the ratio of close (within a specific distance) points among all dynamics' points
    Inputs:
        reco: k X T reconstructed dynamics matrix
        real: k X T real dynamics matrix (ground truth)
        epsilon_close: Threshold for distance
        return_quantiles: whether to return confidence interval values
        quantiles: lower / higher limits for the quantiles
        
    reco: k X T
    real: k X T
    """
    close_enough = np.sqrt(np.sum((reco - real)**2,0)) < epsilon_close

    if return_quantiles:
        try:
            q1,q2 = stats.proportion.proportion_confint(np.sum(close_enough),len(close_enough),quantiles[0])
        except:
            q1 = np.mean(close_enough)
            q2 = np.mean(close_enough)
        return np.mean(close_enough), q1, q2
    return np.mean(close_enough)
    
def plot_bar_contri(contri, ax = [],suptitle = '', fig = []):
    """
    Plot a bar-plot of the values in the dict contri
    Inputs:
        contri   = dictionary whose values are dataframes to plot
        ax       = np.arrays of subplots (its len is the same as the number of keys in contri) (optional)
        suptitle = overall title of all subplots (optional)
        fig      = figure to use (optional)
        
    """
    if isinstance(ax,list):
        if len(ax) == 0:
            fig, ax = plt.subplots(1,len(contri.keys()), figsize = (15,4))
    titles = list(contri.keys())
    [contri[title].plot.bar(ax = ax[i], alpha = 0.6) for i,title in enumerate(titles)]
    [add_labels(ax = ax[i],title = title, xlabel = 'subdyn #', zlabel = None,ylabel = None) for i,title in enumerate(titles)]
    
    # if centralize:
    #     contri_max = np.max([np.max(np.abs(contri[title])) for title in titles])
        
    #     [ax[i].set_ylim([- contri_max ,contri_max ]) for i, title in enumerate(titles)]
    fig.suptitle(suptitle)
    fig.subplots_adjust(wspace = 0.4,hspace = 0.5)

    
def plot_dots_close(reco,real, range_close = [], conf_int = 0.05, ax = [], color = 'blue',label=''):
    """
    For a given reconstructed dynamics ('reco')-> plot a graph of the ratio of dots that are located within a specific distance threshold from the ground truth, as a function of the distance threshold.
    + confidence interval (5%-95%)
    
    Inputs:
        reco        = reconstructed dynamics (the dynamics oobtained by the model). np.array of k X T
        real        = ground-truth dynamics. np.array of k X T.
        range_close = array of possible distances to consider
        conf_int    = confidence interval value (scalar < 0.5)
        ax          = subplot to use (optional)
        color       = color to use (optional)
        label       = curve label
    Output: 
        a np.array of the ratio of 'correct' points, as they are located within the threshold defined by range_close, for each array's index

    """
    if len(range_close) == 0: range_close = np.linspace(10**-8, 10,30)
    if isinstance(ax,list):
        if len(ax) == 0:
            fig, ax = plt.subplots(1,1, figsize = (4,4))    
    vals_close = [claculate_percent_close(reco, real, epsilon_close = close_val, return_quantiles = True, quantiles = [conf_int,1-conf_int]) for close_val in range_close]
    array_close = np.vstack(vals_close)
    ax.plot(range_close, array_close[:,0],color = color, label = label)
    ax.fill_between(range_close, array_close[:,1],array_close [:,2], alpha = 0.2, color = color)
    ax.set_xlabel('Distance')
    ax.set_ylabel('% points')
    return vals_close    

def plot_coefficients_under_speeds(coefficients_mat, dt_range,ax = [] ,min_time = 10000, ax_heat = [], fig = [], fig_heat = [], colors = [], count_plot = 1):
    """"
    Plot the model coefficients obtained under different speeds, as a heatmap. 
    The goal is to compare the coefficients obtained for the same dnamics but with different sampling rates.
    Inputs:
        coefficients_mat   =
        dt_range           = 
        ax                 = subplot to plot the coefficients in
        min_time           = the min. time to plot (left x lim)
        ax_heat            = subplot to draw heatmap
        fig                =  figure to use
        fig_heat           = -||- (for heatmap(
        colors             = colors to use. Shoud be 3 X number_of_different_speeds
        count_plot         =  current # of plot (for labeling)
    """
    if isinstance(ax,list):
        if len(ax) == 0:
            fig,ax = plt.subplots(len(dt_range),1,figsize = (8,10), sharex = True)
    if isinstance(ax_heat,list):
        if len(ax_heat) == 0:
            fig_heat,ax_heat = plt.subplots(2,1,figsize = (5,10), sharex = True)
    if len(colors) == 0:
        len_colors =coefficients_mat.shape[0]
        colors = np.vstack([np.linspace(0,1,len_colors)  ,np.zeros((1,len_colors)), 1-np.linspace(0,1,len_colors)])
    [ax[i].scatter(range(len(coefficients_mat[i,min_time:].T)), coefficients_mat[i,min_time:].T,color = 'b',marker = '*', s=4) for i in range(len(dt_range))] 
    [ax[i].set_title('dt = %.2f'%cur_dt) for i, cur_dt in enumerate(dt_range)]
    ax[0].set_title('c%g \n dt = %.2f'%(count_plot,dt_range[0]))
    ax[-1].set_xlabel('Time')
    try:
        [ax_spec.set_ylim(top = np.nanquantile(coefficients_mat[i,min_time:],0.99)+100) for i,ax_spec in enumerate(ax)]
    except:
        print('Invalid y limit')
    ax_heat[0].plot(pd.DataFrame(coefficients_mat[:,min_time:].T).interpolate(axis = 0).values,lw = 1, alpha = 0.5)
    for i,j in enumerate(ax_heat[0].lines):
        j.set_color(colors[:,i])
   
    ax_heat[0].legend(['dt = %.2f'%cur_dt for i, cur_dt in enumerate(dt_range)], fontsize = 9, loc = 'upper right')
    sns.heatmap(pd.DataFrame(coefficients_mat[:,10000:]).interpolate(axis = 1), cmap = 'PiYG', ax = ax_heat[1])
    add_labels(ax_heat[1], xlabel = 'Time', ylabel = 'Speeds', zlabel = None, yticklabels = dt_range, title = 'c%g'%count_plot )
    add_labels(ax_heat[0], xlabel = 'Time', ylabel = 'Coeffs', zlabel = None, title = 'c%g'%count_plot )
    
    if ~isinstance(fig,list):        fig.subplots_adjust(hspace= 0.83, wspace= 0.4);    fig.suptitle('Coefficients')
    if ~isinstance(fig_heat,list):        fig_heat.subplots_adjust(hspace= 0.23, wspace= 0.4);    fig_heat.suptitle('Coefficients')   
    
    
def load_mat_file(mat_name , mat_path = '',sep = sep):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat','E:\CoDyS-Python-rep-\other_models')
    """
    data_dict = mat73.loadmat(mat_path+sep+mat_name)
    return data_dict




    
    
    
    
    
    
    
def plot_most_likely_dynamics(reg, dynamics_distns,xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
        alpha=0.8,     ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xy.dot(A.T) + b.T - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax
    
    
#%% Plot Multi-colored line
def try_norm_coeffs(coefficients,x_highs_y_highs = [], x_lows_y_lows = [] , choose_meth = 'both',
                            same_width = True,factor_power = 0.9, width_des = 0.7, initial_point = 'start', latent_dyn = [], quarter_initial = 'low'):
    if len(latent_dyn) == 0: raise ValueError('Empty latent dyn was provided')
    coefficients_n = norm_over_time(coefficients, type_norm = 'normal')
    coefficients_n = coefficients_n - np.min(coefficients_n,1).reshape((-1,1))
    if same_width:
        coefficients_n = width_des*(coefficients_n**factor_power) / np.sum(coefficients_n**factor_power,axis = 0)   
    else:
        coefficients_n = coefficients_n / np.sum(coefficients_n,axis = 0)  
    return coefficients_n

    



def plot_weighted_colored_line(dyn, coeffs, ax = [], fig=None ):
    coefficients = norm_over_time(coefficients, type_norm = 'normal')
    if isinstance(ax,list) and len(ax) == 0:
        fig, ax = plt.subplots()
 
            
    
def min_dist(dotA1, dotA2, dotB1, dotB2, num_sects = 500):
    x_lin = np.linspace(dotA1[0], dotA2[0])
    y_lin = np.linspace(dotA1[1], dotA2[1])
    x_lin_or = np.linspace(dotB1[0], dotB2[0])
    y_lin_or = np.linspace(dotB1[1], dotB2[1])
    dist_list = []
    for pairA_num, pairAx in enumerate(x_lin):
        pairAy = y_lin[pairA_num]
        for pairB_num, pairBx in enumerate(x_lin_or):
            pairBy = y_lin_or[pairB_num]
            dist = (pairAx - pairBx)**2 + (pairAy - pairBy)**2
            dist_list.append(dist)
    return dist_list
            
    
    
#%% FHN model
# taken from https://www.normalesup.org/~doulcier/teaching/modeling/excitable_systems.html    
    
def create_FHN(dt = 0.01, max_t = 100, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20, v0 = -0.5, w0 = 0, params = {'exp_power' : 0.9, 'change_speed': False}):
    time_points = np.arange(0, max_t, dt)
    if params['change_speed']:
        time_points = time_points**params['exp_power']
    
        
    w_full = []
    v_full = []
    v = v0
    w = w0
    for t in time_points:
        v, w =  cal_next_FHN(v,w, dt , max_t , I_ext , b, a , tau)
        v_full.append(v)
        w_full.append(w)
    return v_full, w_full


        
def cal_next_FHN(v,w, dt = 0.01, max_t = 300, I_ext = 0.5, b = 0.7, a = 0.8 , tau = 20) :
    v_next = v + dt*(v - (v**3)/3 - w + I_ext)
    w_next = w + dt/tau*(v + a - b*w)
    return v_next, w_next
    
#%% Plot tricolor
def norm_over_time(coefficients, type_norm = 'normal'):
    if type_norm == 'normal':
        coefficients_norm = (coefficients - np.mean(coefficients,1).reshape((-1,1)))/np.std(coefficients, 1).reshape((-1,1))
    return coefficients_norm

def find_perpendicular(d1, d2, perp_length = 1, prev_v = [], next_v = [], ref_point = [],choose_meth = 'intersection',initial_point = 'mid',  
                       direction_initial = 'low', return_unchose = False, layer_num = 0):
    """
    This function find the 2 point of the orthogonal vector to a vector defined by points d1,d2
    d1 =                first data point
    d2 =                second data point
    perp_length =       desired width
    prev_v =            previous value of v. Needed only if choose_meth == 'prev'
    next_v =            next value of v. Needed only if choose_meth == 'prev'
    ref_point =         reference point for the 'smooth' case, or for 2nd+ layers
    choose_meth =       'intersection' (eliminate intersections) OR 'smooth' (smoothing with previous prediction) OR 'prev' (eliminate convexity)
    direction_initial = to which direction take the first perp point  
    return_unchose =    whether to return unchosen directions   
    
    """       
    # Check Input    
    if d2[0] == d1[0] and d2[1] == d1[1]:
        raise ValueError('d1 and d2 are the same point')
    
    # Define start point for un-perp curve
    if initial_point == 'mid':
        perp_begin = (np.array(d1) + np.array(d2))/2
        d1_perp = perp_begin
    elif initial_point == 'end':        d1_perp = d2
    elif initial_point == 'start':        d1_perp = d1
    else:        raise NameError('Unknown intial point')       
    
    # If perpendicular direction is according to 'intersection' elimination
    if choose_meth == 'intersection':
        if len(prev_v) > 0:        intersected_curve1 = prev_v
        else:                      intersected_curve1 = d1
        if len(next_v) > 0:        intersected_curve2 = next_v
        else:                      intersected_curve2 = d2
        
    # If a horizontal line       
    if d2[0] == d1[0]:        d2_perp = np.array([d1_perp[0]+perp_length, d1_perp[1]])
    # If a vertical line
    elif d2[1] == d1[1]:        d2_perp = np.array([d1_perp[0], d1_perp[1]+perp_length])
    else:
        m = (d2[1]-d1[1])/(d2[0]-d1[0]) 
        m_per = -1/m                                       # Slope of perp curve        
        theta1 = np.arctan(m_per)
        theta2 = theta1 + np.pi
        
        # if smoothing
        if choose_meth == 'smooth' or choose_meth == 'intersection':
            if len(ref_point) == 0: 
                #print('no ref point!')
                smooth_val =[]
            else:                smooth_val = np.array(ref_point)
        
        # if by convexity
        if choose_meth == 'prev':
            if len(prev_v) > 0 and len(next_v) > 0:                     # both sides are provided
                prev_mid_or = (np.array(prev_v) + np.array(next_v))/2
            elif len(prev_v) > 0 and len(next_v) == 0:                  # only the previous side is provided
                prev_mid_or = (np.array(prev_v) + np.array(d2))/2
            elif len(next_v) > 0 and len(prev_v) == 0:                  # only the next side is provided               
                prev_mid_or = (np.array(d1) + np.array(next_v))/2
            else:
                raise ValueError('prev or next should be defined (to detect convexity)!')        

        if choose_meth == 'prev':
            prev_mid = prev_mid_or
        elif choose_meth == 'smooth':
            prev_mid = smooth_val
        elif choose_meth == 'intersection':
            prev_mid = smooth_val
            
        x_shift = perp_length * np.cos(theta1)
        y_shift = perp_length * np.sin(theta1)
        d2_perp1 = np.array([d1_perp[0] + x_shift, d1_perp[1]+ y_shift])            
        
        x_shift2 = perp_length * np.cos(theta2)
        y_shift2 = perp_length * np.sin(theta2)
        d2_perp2 = np.array([d1_perp[0] + x_shift2, d1_perp[1]+ y_shift2])
        options_last = [d2_perp1, d2_perp2]
        
        # Choose the option that goes outside
        if len(prev_mid) > 0:
            
          
            if len(ref_point) > 0 and layer_num > 0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((smooth_val - d2_perp1)**2)
                dist2 = np.sum((smooth_val - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])

            elif choose_meth == 'intersection':
                dist1 = np.min(min_dist(prev_mid, d2_perp1, intersected_curve1, intersected_curve2))
                dist2 = np.min(min_dist(prev_mid, d2_perp2, intersected_curve1, intersected_curve2))
                max_opt = np.argmax([dist1,dist2]) 
         
            else:
                dist1 = np.sum((prev_mid - d2_perp1)**2)
                dist2 = np.sum((prev_mid - d2_perp2)**2)
                max_opt = np.argmin([dist1,dist2])  
                     
     
                                 
       
                          
                
        else:
        
            if len(ref_point) > 0 and layer_num >0:                               # here ref point is a point of a different dynamics layer from which we want to take distance
                dist1 = np.sum((ref_point - d2_perp1)**2)
                dist2 = np.sum((ref_point - d2_perp2)**2)
                max_opt = np.argmax([dist1, dist2])
             
            elif direction_initial == 'low':
                max_opt = np.argmin([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'high':
                max_opt = np.argmax([d2_perp1[1], d2_perp2[1]])
            elif direction_initial == 'right' :
                max_opt = np.argmax([d2_perp1[0], d2_perp2[0]])
            elif direction_initial == 'left':
                max_opt = np.argmin([d2_perp1[0], d2_perp2[0]])

                
            else:
                raise NameError('Invalid direction initial value') 
    
    d2_perp = options_last[max_opt] # take the desired direction
    if return_unchose:
        d2_perp_unchose = options_last[np.abs(1 - max_opt)] 
        return d1_perp, d2_perp, d2_perp_unchose
    return d1_perp, d2_perp


def find_lows_high(coeff_row, latent_dyn,   choose_meth ='intersection',factor_power = 0.9, initial_point = 'start',
                   direction_initial = 'low', return_unchose = False, ref_point = [], layer_num = 0):
    """
    Calculates the coordinates of the 'high' values of a specific kayer
    """
    
    if return_unchose: unchosen_highs = []
    ### Initialize
    x_highs_y_highs = []; x_lows_y_lows = []
    if isinstance(ref_point, np.ndarray):
        if len(ref_point.shape) > 1:
            ref_shape_all = ref_point
        else:
            ref_shape_all = np.array([])

    else:
        ref_shape_all = np.array([])
    # Iterate over time
    for t_num in range(0,latent_dyn.shape[1]-2): 
  
        d1_coeff = latent_dyn[:,t_num]
        d2_coeff = latent_dyn[:,t_num+1]
        prev_v = latent_dyn[:,t_num-1] 
        next_v = latent_dyn[:,t_num+2]
        c_len = (coeff_row[t_num] + coeff_row[t_num+1])/2

        if len(ref_shape_all) > 0 and ref_shape_all.shape[0] > t_num and layer_num > 0: # and ref_shape_all.shape[1] >1
            ref_point = ref_shape_all[t_num,:]

          
            if len(ref_point) >  0 and layer_num > 0 :  #and t_num  < 3
                 pass
          
        
        # if do not consider layer
        
        elif t_num > 2 and (choose_meth == 'smooth' or choose_meth == 'intersection'):   
            ref_point  = d2_perp

          
        else:              
            ref_point = []       

        
        if return_unchose:  d1_perp, d2_perp, d2_perp_unchosen = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial =direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        else:               d1_perp, d2_perp = find_perpendicular(d1_coeff, d2_coeff,c_len**factor_power, prev_v = prev_v, next_v=next_v,ref_point  = ref_point , choose_meth = choose_meth, initial_point=initial_point, direction_initial= direction_initial, return_unchose = return_unchose,layer_num=layer_num)# c_len
        # Add results to results lists
        x_lows_y_lows.append([d1_perp[0],d1_perp[1]])
        x_highs_y_highs.append([d2_perp[0],d2_perp[1]])
        if return_unchose: unchosen_highs.append([d2_perp_unchosen[0],d2_perp_unchosen[1]])
    # return
    if return_unchose: 
        return x_lows_y_lows, x_highs_y_highs, unchosen_highs
    return x_lows_y_lows, x_highs_y_highs        


def plot_multi_colors(store_dict,min_time_plot = 0,max_time_plot = -100,  colors = ['green','red','blue'], ax = [],
                      fig = [], alpha = 0.99, smooth_window = 3, factor_power = 0.9, coefficients_n = [], to_scatter = False, 
                      to_scatter_only_one = False ,choose_meth = 'intersection', title = ''):
    """
    store_dict is a dictionary with the high estimation results. 
    example:        
        store_dict , coefficients_n = calculate_high_for_all(coefficients,choose_meth = 'intersection',width_des = width_des, latent_dyn = latent_dyn, direction_initial = direction_initial,factor_power = factor_power, return_unchose=True)
    
    """
    if len(colors) < len(store_dict):                raise ValueError('Not enough colors were provided')
    if isinstance(ax, list) and len(ax) == 0:        fig, ax = plt.subplots(figsize = (20,20))
    for key_counter, (key,set_plot) in enumerate(store_dict.items()):
        if key_counter == 0:
            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1]
            #choose_meth_initial = 
            low_ref =[]
            high_ref = []
        else:
            low_ref = [np.array(x_highs_y_highs)[min_time_plot,0],   np.array(x_highs_y_highs)[min_time_plot,1]]
            high_ref = [np.array(x_highs_y_highs)[max_time_plot,0],np.array(x_highs_y_highs)[max_time_plot,1]]
        if len(coefficients_n) > 0:
            # Define the length of the last perp. 
            c_len = (coefficients_n[key,max_time_plot-1] + coefficients_n[key,max_time_plot])/2
            # Create perp. in the last point            
            d1_p, d2_p =find_perpendicular([np.array(x_lows_y_lows)[max_time_plot-2,0],np.array(x_lows_y_lows)[max_time_plot-2,1]], 
                                           [np.array(x_lows_y_lows)[max_time_plot-1,0],np.array(x_lows_y_lows)[max_time_plot-1,1]], 
                                           perp_length = c_len**factor_power, 
                                           ref_point = high_ref, 
                                           choose_meth = 'intersection',initial_point = 'end')
            # Define the length of the first perp. 
            c_len_start = (coefficients_n[key,min_time_plot-1] + coefficients_n[key,min_time_plot])/2
            # Create perp. in the first point   
            d1_p_start =[np.array(x_highs_y_highs)[min_time_plot,0],np.array(x_highs_y_highs)[min_time_plot,1]]
                                                       
            d2_p_start=  [np.array(x_highs_y_highs)[min_time_plot+1,0],np.array(x_highs_y_highs)[min_time_plot+1,1]]                                                        
            # find_perpendicular([np.array(x_highs_y_highs)[min_time_plot,0],np.array(x_highs_y_highs)[min_time_plot,1]], 
            #                                             [np.array(x_highs_y_highs)[min_time_plot+1,0],np.array(x_highs_y_highs)[min_time_plot+1,1]], 
            #                                             perp_length = c_len**factor_power, 
            #                                             ref_point= low_ref,
            #                                             choose_meth = 'intersection',initial_point = 'start')
            x_lows_y_lows = store_dict[key][0]
            x_highs_y_highs = store_dict[key][1] 
            #d1_p_start = np.array(x_highs_y_highs)[0,0]
            #d2_p_start  = np.array(x_highs_y_highs)[0,1]
            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array([d2_p[0]]), np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,0].flatten(),np.array([d2_p_start[0]])])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array([d2_p[1]]),np.array(x_highs_y_highs)[max_time_plot-1:min_time_plot+1:-1,1].flatten(),np.array([d2_p_start[1]])])
            
        else:
            stack_x = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,0].flatten()])
            stack_y = np.hstack([np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), np.array(x_highs_y_highs)[max_time_plot:min_time_plot:,1].flatten()])
        stack_x_smooth = stack_x
        stack_y_smooth = stack_y
        if key_counter !=0:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=0.3, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#
        else:
            ax.fill(stack_x_smooth, stack_y_smooth, alpha=alpha, facecolor=colors[key_counter], edgecolor=None, zorder=1, snap = True)#

    if to_scatter or (to_scatter_only_one and key == np.max(list(store_dict.keys()))):
        

          ax.scatter(np.array(x_lows_y_lows)[min_time_plot:max_time_plot,0].flatten(), np.array(x_lows_y_lows)[min_time_plot:max_time_plot,1].flatten(), c = 'black', alpha = alpha, s = 45)

    remove_edges(ax)
    if not title  == '':
        ax.set_title(title, fontsize = 20)
    


def remove_edges(ax):
    ax.spines['top'].set_visible(False)    
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def norm_coeffs(coefficients, type_norm, same_width = True,width_des = 0.7,factor_power = 0.9, min_width = 0.01):
    """
    type_norm can be:      'sum_abs', 'norm','abs'
    """
    if type_norm == 'norm':
        coefficients_n =      norm_over_time(np.abs(coefficients), type_norm = 'normal')   
        coefficients_n =      coefficients_n - np.min(coefficients_n,1).reshape((-1,1))
        #plt.plot(coefficients_n.T)
    elif type_norm == 'sum_abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) / np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'abs':
        coefficients[np.abs(coefficients) < min_width] = min_width
        coefficients_n = np.abs(coefficients) #/ np.sum(np.abs(coefficients),1).reshape((-1,1))
    elif type_norm == 'no_norm':
        coefficients_n = coefficients
    else:
        raise NameError('Invalid type_norm value')


    coefficients_n[coefficients_n < min_width]  = min_width
    if same_width:        coefficients_n = width_des*(np.abs(coefficients_n)**factor_power) / np.sum(np.abs(coefficients_n)**factor_power,axis = 0)   
    else:                 coefficients_n = np.abs(coefficients_n) / np.sum(np.abs(coefficients_n),axis = 0)  
    coefficients_n[coefficients_n < min_width]  = min_width
    return coefficients_n

    
def calculate_high_for_all(coefficients, choose_meth = 'both', same_width = True,factor_power = 0.9, width_des = 0.7, 
                           initial_point = 'start', latent_dyn = [],
                          direction_initial = 'low', return_unchose = False, type_norm = 'norm',min_width =0.01):
    """
    Create the dictionary to store results
    """
    if len(latent_dyn) == 0: raise ValueError('Empty latent dyn was provided')
    
    # Coeffs normalization
    coefficients_n = norm_coeffs(coefficients, type_norm, same_width = same_width, width_des = width_des,factor_power =factor_power,min_width=min_width )
    
    # Initialization
    store_dict      = {}
    dyn_use         = latent_dyn
    ref_point       = []
    
    for row in range(coefficients_n.shape[0]):
        #print(row)
        coeff_row = coefficients_n[row,:]
        # Store the results for each layer
        if return_unchose:
            x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2 = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                                             initial_point = initial_point,direction_initial = direction_initial,
                                                                             return_unchose = return_unchose, ref_point = ref_point,layer_num = row )             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs,x_highs_y_highs2]
        else:
            x_lows_y_lows, x_highs_y_highs = find_lows_high(coeff_row,dyn_use, choose_meth = choose_meth, factor_power=factor_power, 
                                                            initial_point = initial_point, direction_initial = direction_initial ,
                                                            return_unchose = return_unchose,ref_point = ref_point ,layer_num=row)             
            store_dict[row] = [x_lows_y_lows, x_highs_y_highs]
        # Update the reference points    
        if initial_point == 'mid':
            dyn_use = np.array(x_highs_y_highs).T
            dyn_use = (dyn_use[:,1:] + dyn_use[:,:-1])/2
            dyn_use = np.hstack([latent_dyn[:,:2], dyn_use, latent_dyn[:,-2:]])
        else:
            dyn_use = np.array(x_highs_y_highs).T
        #if row > 0:
        ref_point = np.array(x_lows_y_lows)#[0,:]
        #print('ref point')
        #ref_point)
        #else:
        #   ref_point = []
    return store_dict, coefficients_n    

def add_bar_dynamics(coefficients_n, ax_all_all = [],min_max_points = [10,100,200,300,400,500], 
                     colors = np.array(['r','g','b','yellow']), centralize = False):
    if isinstance(ax_all_all, list) and len(ax_all_all) == 0:
        fig, ax_all_all  = plt.subplots(1,len(min_max_points), figsize = (8*len(min_max_points), 7))

    max_bar = np.max(np.abs(coefficients_n[:,min_max_points]))
    for pair_num,val in enumerate(min_max_points):
        ax_all = ax_all_all[pair_num]

        
        ax_all.bar(np.arange(coefficients_n.shape[0]),coefficients_n[:,val], 
                   color = np.array(colors)[:coefficients_n.shape[0]],
                   alpha = 0.3)
        # ax_all.set_title('t = %s'%str(val), fontsize = 40, fontweight = 'bold')
        ax_all.get_xaxis().set_ticks([]) #for ax in ax_all]
        ax_all.get_yaxis().set_ticks([]) #for ax in ax_all]
        ax_all.spines['top'].set_visible(False)
        
        ax_all.spines['right'].set_visible(False)
        ax_all.spines['bottom'].set_visible(False)
        ax_all.spines['left'].set_visible(False)  
        ax_all.axhline(0, ls = '-',alpha = 0.5, color = 'black', lw = 6)
        ax_all.set_ylim([-max_bar,max_bar])
def create_name_fhn_files(reg_value, num_dynamics):
    new_name = r'E:\CoDyS-Python-rep-\fhn\multifhn_%ssub%sreg.npy'%(str(num_dynamics), str(reg_value).replace('.','_'))
    return new_name
    
#%% Plot 2d axis of coeffs for fig 2

def plot_3d_dyn_basis(F, coefficients, projection = [0,-1], ax = [],  fig = [], time_emph = [], n_times = 5,
                      type_plot = 'quiver',range_p = 10,s=200, w = 0.05/3, alpha0 = 0.3, 
                      time_emph_text  = [10, 20, 30, 50,80, 100,200,300,400,500], turn_off_back = True, lim1 = np.nan, 
                      ax_qui = [], 
                      ax_base = [], to_title = True, loc_title = 'title', include_bar =True, axs_basis_colored = [],
                      colors_dyns = np.array(['r','g','b','yellow']) , plot_dyn_by_colorbase = False, remove_edges_ax = False, include_dynamics = False,
                      latent_dyn = [],fontsize_times = 16,delta_text = 0.1, delta_text_y = 0,delta_text_z = 0, 
                      new_colors = True, include_quiver = True, base_narrow = True,colors = [],color_by_dom = False,
                      quiver_3d = False, s_all = 10,to_remove_edge = True, to_grid = False, cons_color = False):   
    """
    ax = subplot to plot coefficients over time
    colors = should be a mat of k X 3
    """
    if not F[0].shape[0] ==3: quiver_3d = False
    if len(colors) ==0:    
        if color_by_dom:
            color_sig_tmp = find_dominant_dyn(np.abs(coefficients))
            colors = colors_dyns[color_sig_tmp]
            colors_base = np.zeros(coefficients.shape[1])
            
        else:
            colors_base = np.linspace(0,1,coefficients.shape[1]).reshape((-1,1))    
            colors = np.hstack([colors_base, 1-colors_base, colors_base**2])    

    if isinstance(ax,list) and len(ax) == 0:
        if len(F) == 3:        fig, ax = plt.subplots(subplot_kw={'projection':'3d'}, figsize= (10,10))
        elif len(F) == 2:      fig, ax = plt.subplots(figsize= (10,10))
        else: raise ValueError('Invalid dim for F')
    if len(time_emph) == 0: 
        time_emph =np.linspace(0,coefficients.shape[1]-2, n_times+1)[1:].astype(int)

    if include_dynamics:
        
        if len(latent_dyn) == 0: raise ValueError('You should provide latent dyn as input if "include dynamics" it True')
        if len(F[0]) == 3:
            fig_dyn,ax_dyn = plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
            if new_colors:

                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)],latent_dyn[2,:len(colors_base)], color = colors,alpha = 0.3)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = 'black', s = 300)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:],latent_dyn[2,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph],latent_dyn[2,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y,latent_dyn[2,t]+delta_text_z, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            ax_dyn.set_axis_off()
            
        else:
            fig_dyn,ax_dyn = plt.subplots(figsize = (10,10))
            if new_colors:
                
                
                ax_dyn.scatter(latent_dyn[0,:len(colors_base)], latent_dyn[1,:len(colors_base)], color = colors,alpha = 0.3, s = 50)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = 'black', s = 200)
            else:
                c_sig = np.arange(latent_dyn.shape[1])
                ax_dyn.scatter(latent_dyn[0,:], latent_dyn[1,:], c = c_sig,alpha = 0.3, cmap = 'viridis', s = 100)
                ax_dyn.scatter(latent_dyn[0,time_emph],latent_dyn[1,time_emph], c = c_sig[time_emph], s = 300, cmap = 'viridis')
            [ax_dyn.text(latent_dyn[0,t] + delta_text,latent_dyn[1,t]+delta_text_y, 't = %s'%str(t), fontsize =fontsize_times, fontweight = 'bold') for t in time_emph]
            remove_edges(ax_dyn)
            #ax_dyn.set_axis_off()
    if len(F[0]) == 3: 

        if quiver_3d:
            if type_plot == 'streamplot': 
                type_plot = 'quiver'
                print('If quiver_3d then type_plot need to be quiver (currently is streamplot)')
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         
                    fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) ,subplot_kw={'projection':'3d'})
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) ,subplot_kw={'projection':'3d'})
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ),subplot_kw={'projection':'3d'})
        else:
          
            F = [f[:,projection] for f in F]
            F = [f[projection, :] for f in F]
            if  include_quiver:
                if isinstance(ax_qui, list) and len(ax_qui)== 0:         fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5) )
            if isinstance(ax_base, list) and len(ax_base)==0:         
                if base_narrow:
                    fig_base, ax_base = plt.subplots(len(F),1, figsize= (5,7*len(F)) )
                else:
                    fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))

    elif len(F[0]) == 2:  
        if isinstance(ax_qui, list) and len(ax_qui)==0:     fig_qui, ax_qui = plt.subplots(1,len(time_emph), figsize= (7*len(time_emph),5))
        if isinstance(ax_base, list) and len(ax_base)==0:   
            if base_narrow:
                fig_base, ax_base = plt.subplots(len(F), 1,figsize= (5,7*len(F) ))
            else:
                fig_base, ax_base = plt.subplots(1,len(F),figsize= (7*len(F),5 ))
    if len(F[0]) == 3:

        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:

            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors, alpha = alpha0, s = s_all)

            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black',
                       s = s)

            [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
        else:
            
            cmap = matplotlib.cm.get_cmap('viridis')
            colors_base = np.arange(coefficients.shape[1])

            ax.scatter(coefficients[0,:],coefficients[1,:],coefficients[2,:], c = colors_base, alpha = alpha0, s = s_all)

            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph],coefficients[2,time_emph], c = 'black', s = s, alpha = np.min([alpha0*2, 1]))
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p,  
                           color = cmap(time_point/colors.shape[0])  ,
                       w = w, ax = ax_qui[i], quiver_3d = quiver_3d, cons_color = cons_color) for i, time_point in enumerate(time_emph)]
            
        if to_title and include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize =fontsize_times*3 , fontweight = 'bold') for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for i, time_point in enumerate(time_emph)]

        [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point), fontsize =fontsize_times, fontweight = 'bold') for time_point in time_emph_text]
        ax.set_xlabel('f1');ax.set_ylabel('f2');ax.set_zlabel('f3');

    else:
        


        cmap = matplotlib.cm.get_cmap('viridis')
        if new_colors:
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color =colors[time_point] , w = w, ax = ax_qui[i],cons_color=cons_color ) for i, time_point in enumerate(time_emph)]
        else:
            colors_base = np.arange(coefficients.shape[1])
            ax.scatter(coefficients[0,:],coefficients[1,:], c = colors_base, alpha = alpha0, s = s_all)
            ax.scatter(coefficients[0,time_emph],coefficients[1,time_emph], c = colors_base[time_emph], s = s)
            if include_quiver:
                [plot_reco_dyn(coefficients, F, time_point, type_plot = type_plot, range_p = range_p, color = cmap(time_point/colors.shape[0]) , w = w, ax = ax_qui[i],cons_color= cons_color ) for i, time_point in enumerate(time_emph)]
        if latent_dyn.shape[0] == 3:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y, coefficients[2,time_point]+delta_text_z,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
    
        else:
            [ax.text(coefficients[0,time_point]+delta_text,coefficients[1,time_point]+delta_text_y,'t = ' + str(time_point),fontsize = fontsize_times ,fontweight = 'bold') for time_point in time_emph_text]
        
        ax.set_xlabel('f1');ax.set_ylabel('f2');
        if  remove_edges_ax:        remove_edges(ax)

        if to_title and  include_quiver:
            if loc_title == 'title':
                [ax_qui[i].set_title('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
            else:
                [ax_qui[i].set_ylabel('t = ' + str(time_point), fontsize = 30) for i, time_point in enumerate(time_emph)]
    if to_remove_edge:  
        if  include_quiver:        [remove_edges(ax_spec) for ax_spec in ax_qui]
        [remove_edges(ax_spec) for ax_spec in ax_base]
        ax.set_xticks([])
        ax.set_yticks([])
        if quiver_3d:
            ax.set_zticks([])    
    #if include_quiver:
    [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = ax_base[f_num],chosen_color =  'black', w = w, type_plot = type_plot,cons_color =cons_color,quiver_3d = quiver_3d ) for f_num, f in enumerate(F)]
    [ax_base_spec.set_title('f %s'%str(i), fontsize = 16) for i, ax_base_spec in enumerate(ax_base)]
    

    if turn_off_back and  len(F) == 3:
      ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
      ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not to_grid and  len(F) == 3:
       
      ax.grid(False)
      # Hide axes ticks

      ax.set_zticks([])
      
      ax.xaxis._axinfo['juggled'] = (0,0,0)
      ax.yaxis._axinfo['juggled'] = (1,1,1)
      ax.zaxis._axinfo['juggled'] = (2,2,2)
    if not np.isnan(lim1):
        ax.set_xlim([-lim1,lim1])
        ax.set_ylim([-lim1,lim1])
        
        
    if include_bar:
        if base_narrow:
            fig_all_all, ax_all_all = plt.subplots(len(time_emph),1, figsize = (6,len(time_emph)*7))
        else:
            ax_all_all = []
        add_bar_dynamics(coefficients, ax_all_all = ax_all_all, min_max_points = time_emph, colors = colors_dyns, 
                         centralize = True)

        if isinstance( axs_basis_colored ,list) and len( axs_basis_colored ) == 0:
            if base_narrow:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( len(F),1,figsize = (5,6*len(F)))
                
            else:
                if quiver_3d: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5),subplot_kw={'projection':'3d'})
                else: fig_basis_colored , axs_basis_colored = plt.subplots( 1, len(F), figsize = (6*len(F),5))
        [quiver_plot(f,-range_p, range_p, -range_p, range_p, ax = axs_basis_colored[f_num],alpha = 0.7, chosen_color =  colors_dyns[f_num], w = w, type_plot = type_plot, cons_color = cons_color, quiver_3d=quiver_3d ) for f_num, f in enumerate(F)]
        [remove_edges(ax_spec) for ax_spec in axs_basis_colored]
        if quiver_3d:        [ax.set_zticks([]) for ax in axs_basis_colored]
       

            
def plot_reco_dyn(coefficients, F, time_point, type_plot = 'quiver', range_p = 10, color = 'black',
                  w = 0.05/3, ax = [], cons_color= False, to_remove_edges = False, projection = [0,1], 
                  return_artist = False,
                  xlabel = 'x',ylabel = 'y',quiver_3d = False):
    if isinstance(ax,list) and len(ax) == 0:
  
        fig, ax = plt.subplots()

    if len(F) == 3:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]+coefficients[2,time_point] * F[2]

        if not quiver_3d:      

            merge_dyn_at_t_break = merge_dyn_at_t_break[:, projection]
            merge_dyn_at_t_break = merge_dyn_at_t_break[projection,:]

    elif len(F) == 2:
        merge_dyn_at_t_break = coefficients[0,time_point] * F[0]+coefficients[1,time_point] * F[1]

    art = quiver_plot(sub_dyn = merge_dyn_at_t_break, chosen_color = color,  xmin = -range_p, 
                      xmax = range_p, ymin= -range_p,ymax= range_p, ax = ax, w = w, type_plot=type_plot,
                      cons_color= cons_color, return_artist = return_artist, xlabel = xlabel, ylabel = ylabel,
                      quiver_3d = quiver_3d)
    if to_remove_edges: remove_edges(ax)
    if return_artist:
        return art



def plot_c_space(coefficients,latent_dyn = [], axs = [], fig = [], xlim = [-50,50], ylim = [-50,50], add_midline = True, d3 = True, cmap = 'winter', color_sig = [], 
                 title = '', times_plot= [], cmap_f = []):
    if len(times_plot) > 0 and isinstance(cmap_f, list): cmap_f = plt.cm.get_cmap(cmap)
    if len(color_sig) == 0:    color_sig = latent_dyn[0,:-1]
    if isinstance(axs, list) and len(axs) == 0:
      
        if coefficients.shape[0] == 3:
            fig, axs =  plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
            d3 = True
            h = axs.scatter(coefficients[0,:], coefficients[1,:],coefficients[2,:], c = color_sig, cmap = cmap)
            if len(times_plot) > 0:
                axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], coefficients[2,times_plot], c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
        elif coefficients.shape[0] == 2:
            if d3:       
                fig, axs =  plt.subplots(figsize = (15,15),subplot_kw={'projection':'3d'})
                h = axs.scatter(coefficients[0,:], coefficients[1,:], np.arange(coefficients.shape[1]), c = color_sig, cmap = cmap)
                if len(times_plot) > 0:
                    zax = np.arange(coefficients.shape[1])
                    axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot], zax[times_plot], c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
            else:
                fig, axs =  plt.subplots(figsize = (15,15))
                h = axs.scatter(coefficients[0,:], coefficients[1,:], c = color_sig, cmap = cmap)
                if len(times_plot) > 0:
                    axs.scatter(coefficients[0,times_plot], coefficients[1,times_plot],  c =cmap_f(color_sig[times_plot]/np.max(color_sig)),s = 500 )
        else:
            print('Invalid coefficients shape in axis 0')
    
    if len(xlim) > 0:    axs.set_xlim(xlim)
    if len(ylim) > 0:    axs.set_ylim(ylim)
    if not isinstance(fig, list):    fig.colorbar(h)

    if d3: 
        axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.grid(False)
        axs.set_axis_off()
    if add_midline:
        if d3:
            axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axs.plot([0,0],[np.min(coefficients[1,:]),np.max(coefficients[1,:])],[0,0], color = 'black', ls = '--', alpha = 0.3)
            axs.plot([np.min(coefficients[0,:]),np.nanmax(coefficients[0,:])],[0,0],[0,0], color = 'black', ls = '--', alpha = 0.3)
            axs.plot([0,0],[0,0],[0,1.3*coefficients.shape[1]],color = 'black', alpha = 0.3, ls = '--')
            axs.view_init(elev=15, azim=30)
        else:
            axs.axhline(0, color = 'black', ls = '--', alpha = 0.3)
            axs.axvline(0, color = 'black', ls = '--', alpha = 0.3)
    if len(title) > 0:
        axs.set_title(title)
    
    
    


    
    
    
    

        