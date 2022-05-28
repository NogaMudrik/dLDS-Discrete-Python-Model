# -*- coding: utf-8 -*-

from importlib import reload  

import main_functions
main_functions = reload(main_functions)
from main_functions import * 
import create_params
from datetime import date



"""
Parameters
"""
exec(open('create_params.py').read())
download_py = False
max_dyn =5
dyn_radius = 5
dyn_num_cyls = 3
addi_save = date.today().strftime('%d%m%y') # For saving
return_evolution  = False;                  # whether to save intermediate results
if 'addition_save' not in locals():    addition_save = []
start_sparse_c = False
init_distant_F = True
max_corr = 0.1
GD_decay = 0.99
smooth_term = 0
min_dyn = 2
update_c_types =['spgl1'] 
num_iters =[10]

max_iter = 6000
params_ex = {'radius':10, 'num_cyls': 5, 'bias':0,'exp_power':0.2}
noise = 0
speed_change = 0
is_D_I = True
"""
Parameters to choose
"""
dt = float(input('dt (rec for Lorenz 0.01, rec for FHN 0.2)'))
max_time = float(input('max time (rec for Lorenz 10, rec for FHN 200)'))
dynamic_type = input('dynamic type (e.g. lorenz, FHN)')#'lorenz' #input('dyn type?');#'circ2d'#input('dynamic_type?cyl/multi_cyl..')# 'cyl'
addi_name = input('additional name id')
num_subdyns = [int(input('num_dyns (m)'))]
include_last_up = str2bool(input('include last up? (for FHN reg)'))
reg_vals_new = [float(input('reg_val_input (tau)'))]
addition_save.append(addi_save)



"""
Runnining over the parameters
"""


for num_iter in num_iters:
    for reg_value in reg_vals_new:     
        for update_c_type in update_c_types :
                for num_subs in num_subdyns:
                    
                    step_f = 50
                    step_f_original = step_f                   
                    
                    to_save_without_ask = True
                    num_subdyns = num_subs
                    
                    reg_term = reg_value
                    max_error =  1e-8 
                    max_data_reco = max_error
                    acumulated_error = False
                    error_order_initial = 2
                    error_step_add = 120
                    
                    error_step_max =2
                    error_step_max_display =2
                    error_step_add = 120
                    epsilon_error_change= 10**(-8)
                    sigma_mix_f = 0.1
                    weights_orders = np.linspace(1,2**error_step_max,error_step_max)[::-1]
                    grad_vec = np.linspace(0.99,1.01,error_step_max)
                    
                    action_along_time = 'median'  #can also be 'mean'
                    
                    latent_dyn            = create_dynamics(type_dyn = dynamic_type, max_time = max_time, dt = dt)
                    F                     = [init_mat((latent_dyn.shape[0], latent_dyn.shape[0]),normalize=True) for i in range(num_subdyns)]
                    coefficients          = init_mat((num_subdyns,latent_dyn.shape[1]-1))
                    include_D             = False
                    
                    error_reco            = np.inf
                    error_reco_all        = error_reco_all = np.inf*np.ones((1,error_step_max))
                    cur_reco              = create_reco(latent_dyn=latent_dyn, coefficients= coefficients, F=F, accumulation = acumulated_error)
                    error_reco_array      = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
                    error_reco_array_med  = np.inf*np.ones((1,max(error_step_max_display,error_step_max)))
                    error_order           = error_step_max 
                    name_auto = True
                    save_name = '%s_%gsub%greg%s_iters%s'%(dynamic_type,num_subdyns,reg_term, update_c_type, str(num_iter))
                    
                    seed_f                = 0
                    dt_range              = np.linspace(0.001, 1, 20)
                    exp_power             = 0.1
                    data                  = []
                    clean_dyn             = []
                    normalize_eig         = True
                    same_c                = False
                    download_py           = False
                    to_print              = False
            
                    params_ex = {'radius':dyn_radius , 'num_cyls': dyn_num_cyls, 'bias':dyn_bias,'exp_power':exp_power}
                
                    params_update_c = {'reg_term': reg_term, 'update_c_type':update_c_type,'smooth_term' :smooth_term, 'num_iters': num_iter, 'threshkind':'soft'}
                    
                    if  to_save_without_ask:
                        to_load = False
                    else:
                        to_load = str2bool(input('To load?'))
                    if to_load:
                        file_name = input('file_name (tok if you want to upload)')
                        if file_name == 'tok':
                            filename = askopenfilename()
                    
                        iter_results = np.load(filename,allow_pickle=True).item()
                        F = iter_results['F']
                        coefficients = iter_results['coefficients']   
                        latent_dyn = iter_results['latent_dyn']
                        if 'error_reco_array' in iter_results: error_reco_array = iter_results['error_reco_array']
                        if 'error_reco_array_med' in iter_results: error_reco_array_med = iter_results['error_reco_array_med']
                        if 'max_time' in iter_results:    max_time = iter_results['max_time']   
                        if 'dt' in iter_results:  dt = iter_results['dt']
                        if 'dyn_type' in iter_results: dynamic_type = iter_results['dyn_type']
                        if 'params_ex' in iter_results: params_ex = iter_results['params_ex']        
                    
                    
                    else:  
                        if to_save_without_ask: to_save = True
                        else: to_save = str2bool(input('To save?'))
                            
                        if to_save: 
                            if name_auto:
                                pass#save_name 
                            else:
                                save_name = input('save_name')
                        if return_evolution:
                            coefficients, F, data, error_reco_array, error_reco_array_med,D , store_iter_restuls,additional_return= train_model_include_D(max_time , dt ,  dynamic_type, num_subdyns = num_subdyns, error_reco = error_reco
                                                                                                                    ,error_step_max = error_step_max, error_order = error_order , data = data ,same_c = same_c, 
                                                                                                                    step_f = step_f, GD_decay =  GD_decay ,   weights_orders = weights_orders ,clean_dyn = clean_dyn ,max_error = max_error ,grad_vec = grad_vec , max_iter = max_iter, include_D=include_D, seed_f = seed_f, normalize_eig = normalize_eig, params_ex = params_ex, to_print = to_print,params = params_update_c, return_evolution  = return_evolution )
                        else:
                            
                            coefficients, F, data, error_reco_array, error_reco_array_med,D,additional_return= train_model_include_D(max_time , dt ,  dynamic_type, num_subdyns = num_subdyns, error_reco = error_reco
                                                                                                                    ,error_step_max = error_step_max, error_order = error_order , data = data ,same_c = same_c, 
                                                                                                                    step_f = step_f, GD_decay =  GD_decay ,   weights_orders = weights_orders ,clean_dyn = clean_dyn ,max_error = max_error ,grad_vec = grad_vec , max_iter = max_iter, include_D=include_D, seed_f = seed_f, normalize_eig = normalize_eig, params_ex = params_ex, to_print = to_print,params = params_update_c, return_evolution  = return_evolution )
                
   
                        if isinstance(data, list) and len(data) > 0:
                    
                            latent_dyn = latent_dyn_base
                        else:
                            latent_dyn  = data
                        
                        # Save Results
                        if not to_load:
                            
                            if to_save: 
                                if not return_evolution:
                                    save_dict = {'F':F, 'coefficients':coefficients, 'latent_dyn': latent_dyn, 'max_time': max_time, 'dt':dt,'dyn_type':dynamic_type,'error_reco_array' :error_reco_array, 'error_reco_array_med' : error_reco_array_med,'params_ex':params_ex}          
                                else:
                                    save_dict = {'F':F, 'coefficients':coefficients, 'latent_dyn': latent_dyn, 'max_time': max_time, 'dt':dt,'dyn_type':dynamic_type,'error_reco_array' :error_reco_array, 'error_reco_array_med' : error_reco_array_med,'params_ex':params_ex,'store_iter_restuls':store_iter_restuls}          
                                    
                                    
                                save_file_dynamics(save_name, ['main_folder_results', dynamic_type, 'clean%s'%addi_name,update_c_type ]+addition_save, save_dict )
                                save_file_dynamics(save_name, ['main_folder_results' ,dynamic_type, 'clean%s'%addi_name,update_c_type ]+addition_save, [], type_save = '.pkl' )
                
