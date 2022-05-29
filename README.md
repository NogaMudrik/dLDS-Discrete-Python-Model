# Discrete-Python-Model---Decomposed-Linear-Dynamical-Systems-dLDS-paper
Decomposed Linear Dynamical Systems (dLDS) for \newline  learning the latent components of neural dynamics


# Package Explanation
## Functions:
##### 1. train_model_include_D:
_main function to train the model._

**train_model_include_D**_(max_time = 500, dt = 0.1, dynamics_type = 'cyl',num_subdyns = 3, 
                          error_reco = np.inf,  data = [], step_f = 30, GD_decay = 0.85, max_error = 1e-3, 
                          max_iter = 3000, F = [], coefficients = [], params= {'update_c_type':'inv','reg_term':0,'smooth_term':0}, 
                          epsilon_error_change = 10**(-5), D = [], x_former =[], latent_dim = None, include_D  = False,step_D = 30, reg1=0,reg_f =0 , 
                          max_data_reco = 1e-3,  sigma_mix_f = 0.1,  action_along_time = 'median', to_print = True, seed = 0, seed_f = 0, 
                          normalize_eig  = True,  params_ex = {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}, 
                          init_distant_F = False,max_corr = 0.1, decaying_reg = 1, other_params_c={}, include_last_up = False)_

#### Parameters:
      max_time         = Number of time points for the dynamics. Relevant only if data is empty;
      dt               =  time interval for the dynamics
      dynamics_type    = type of the dynamics. Can be 'cyl', 'lorenz','FHN', 'multi_cyl', 'torus', 'circ2d', 'spiral'
      num_subdyns      = number of sub-dynamics
      error_reco       = intial error for the reconstruction (do not touch)
      data             = if one wants to use a pre define groud-truth dynamics. If not empty - it overwrites max_time, dt, and dynamics_type
      step_f           = initial step size for GD on the sub-dynamics
      GD_decay         = Gradient descent decay rate
      max_error        = Threshold for the model error. If the model arrives at a lower reconstruction error - the training ends.
      max_iter         = # of max. iterations for training the model
      F                = pre-defined sub-dynamics. Keep empty if random.
      coefficients     = pre-defined coefficients. Keep empty if random.
      params           = dictionary that includes info about the regularization and coefficients solver. e.g. {'update_c_type':'inv','reg_term':0,'smooth_term':0}
      epsilon_error_change = check if the sub-dynamics do not change by at least epsilon_error_change, for at least 5 last iterations. Otherwise - add noise to f
      D                = pre-defined D matrix (keep empty if D = I)
      latent_dim       =  If D != I, it is the pre-defined latent dynamics.
      include_D        = If True -> D !=I; If False -> D = I
      step_D           = GD step for updating D, only if include_D is true
      reg1             = if include_D is true -> L1 regularization on D
      reg_f            = if include_D is true ->  Frobenius norm regularization on D
      max_data_reco    = if include_D is true -> threshold for the error on the reconstruction of the data (continue training if the error (y - Dx)^2 > max_data_reco)
      sigma_mix_f            = std of noise added to mix f
      action_along_time      = the function to take on the error over time. Can be 'median' or 'mean'
      to_print               = to print error value while training? (boolean)
      seed                   = random seed
      seed_f                 = random seed for initializing f
      normalize_eig          = whether to normalize each sub-dynamic by dividing by the highest abs eval
      params_ex              = parameters related to the creation of the ground truth dynamics. e.g. {'radius':1, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
      init_distant_F         = when initializing F -> make sure that the correlation between each pair of {f}_i does not exeed a threshold
      max_corr               = max correlation between each pair of initial sub-dyns (relevant only if init_distant_F is True)
      decaying_reg           = decaying factor for the l1 regularization on the coefficients. If 1 - there is no decay. (should be a scalar in (0,1])
      other_params_c         = additional parameters for the update step of c
      include_last_up        = add another update step of the coefficients at the end
      
* example call (for Lorenz, w. 3 operators):       train_model_include_D(10, 0.01, 'lorenz', 3, GD_decay = 0.99)
    
