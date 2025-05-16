import astropy.units as u
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time, TimeDelta
#import zodipy
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import PIL
import multiprocessing
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from IPython import display
from cosmoglobe.tod_tools import TODLoader
import jax
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm
import corner
import time
from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import ternary
import mpltern
import sys
sys.path.insert(0, "/uio/hume/student-u59/taabeled/Desktop/Masters thesis/Masters/Zodipy_Thomas/zodipy_Thomas/")
import zodipy
from zodipy import model_registry
from astropy import units
from jax.test_util import check_grads
from functools import partial
#from scipy.optimize import minimize
import cProfile
import copy
import ast
from jax.scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
np.random.seed(5)
t0_timer = time.time()

#JAX doesn't like having text sent in so have to make a function that converts string into numbers and numbers into string
#String_list shoudl look like this 
# EX.1 ["comps", "cloud", ["Omega", "beta"]]
# Ex.2 ["comps", "cloud"] This will change all the parameters within parameter["comps"]["cloud"]
# note: Are going to change so it's sent as 
# [["comps", "spectrum"], [["cloud", "bands"], ["clouds", "bands"]], [ [["Omega", "beta"], ["Omega"]] , [["Omega"], ["Omega"]] ]
# So it'll be like [2, [5, 5], [ 5, 5, 5] ] 
def parameter_list_str_to_nmbr(string_list, model):
    parameters = model.get_parameters()
    top = list(parameters.keys())
    #['comps', 'spectrum', 'T_0', 'delta', 'emissivities', 'albedos', 'solar_irradiance', 'C1', 'C2', 'C3']
    #Top has 10 parameters

    #middle = list(parameters["comps"].keys())
    #bottom = list(parameters["comps"]["cloud"].keys())
    #print(f"string_list[2] = {string_list[2][0][0][0]}")
    #print(f"string_list = {string_list}")
    top_arr = np.zeros(len(string_list[0]))
    flat_string_list_1 = flatten_list(string_list[1])
    flat_middle_arr = np.zeros(len(flat_string_list_1))
    middle_arr = restructure_flattend_list(flat_middle_arr, string_list[1])
    flat_string_list_2 = flatten_list(string_list[2])
    flat_bottom_arr = np.zeros(len(flat_string_list_2))
    #print(f"string_list[2] = {string_list[2]}")
    #print(f"flat_bottom_arr = {flat_bottom_arr}")
    bottom_arr = restructure_flattend_list(flat_bottom_arr, string_list[2])
    for i in range(len(string_list[0])):
        if string_list[0][i] in top:
            top_index = top.index(string_list[0][i])
            top_arr[i] = int(top_index)

            #check if it has values and not keys
            if hasattr(parameters[top[top_index]], "keys"):
                middle = list(parameters[top[top_index]].keys())
            else:
                #print(f"parameters[string_list[0][i]] = {parameters[string_list[0][i]]}")
                middle_arr[i] = parameters[string_list[0][i]] #Let the middle array contain the tuple/array/value
                bottom_arr = False
                continue

            for j in range(len(string_list[1])):
                if string_list[1][i][j] in middle:
                    middle_index = middle.index(string_list[1][i][j])
                    middle_arr[i][j] = int(middle_index)
                    if len(string_list[2][i][j]) > 0:
                        #bottom_arr = np.zeros_like(string_list[2])
                        #check if it has a tuple/array of values
                        if hasattr(parameters[top[top_index]][middle[middle_index]], "keys"):
                            bottom = list(parameters[top[top_index]][middle[middle_index]].keys())
                        else:
                            bottom_arr[i][j] = parameters[string_list[0][j]][string_list[1][i][j]] #Let the bottom array contain the tuple/array
                            continue

                        for k in range(len(string_list[2][i][j])):
                            if string_list[2][i][j][k] in bottom:
                                bottom_index = bottom.index(string_list[2][i][j][k])
                                bottom_arr[i][j][k] = int(bottom_index)
                            else:
                                print(f"Did not find {string_list[2][i][j][k]} in bot")
                    else:
                        bottom_arr = [[[]]]
                        #check if it has a tuple of values
                        #print(f"parameters[string_list[0][i]][string_list[1][i][j]] = {parameters[string_list[0][i]][string_list[1][i][j]]}")
                        if hasattr(parameters[top[top_index]][middle[middle_index]], "keys"):
                            bottom = list(parameters[top[top_index]][middle[middle_index]].keys())
                            #print(f"Working as inteded 2")
                        else:
                            #print(f"parameters[string_list[0][i]][string_list[1][i][j]] = {parameters[string_list[0][i]][string_list[1][i][j]]}")
                            bottom_arr[i][j] = parameters[string_list[0][i]][string_list[1][i][j]] #Let the bottom array contain the tuple/array
                            #print(f"bottom_arr = {bottom_arr}")
                            continue
                        bottom_arr[i][j] = list(np.arange(0, len(bottom), dtype=int))
                else:
                    print(f"Did not find {string_list[1][i][j]} in mid")
        else: 
            print(f"Did not find {string_list[0][i]} in top")
    top_arr = np.array(top_arr, dtype=int)
    middle_arr = np.array(middle_arr, dtype=object)
    if bottom_arr == False:
        bottom_arr = [[[]]]
        number_list = [top_arr, middle_arr, bottom_arr]
    else:
        #print(f"bottom_arr = {len(bottom_arr[0][0])}")
        bottom_arr = np.array(bottom_arr, dtype=object)
        #print(f"bottom_arr = {len(bottom_arr[0][0])}")
        
        number_list = [top_arr, middle_arr, bottom_arr]

    #print(f"number_list = {number_list}")
    return number_list

#Function that transalates parameter numbers into strings
def parameter_nmbr_to_str_list(number_array, model):
    parameters = model.get_parameters()
    #Top
    top = list(parameters.keys())
    top_str_arr = []
    middle_str_arr_0 = []
    bottom_str_arr_0 = []
    #print(f"number_array = {number_array}")
    for i in range(len(number_array[0])):
        #print(f"number_array[0][i] = {number_array[0][i]}")
        top_str = top[number_array[0][i]]
        top_str_arr.append(top_str)
        #Middle
        middle_str_arr_1 = []
        bottom_str_arr_1 = []
        #print(f"parameters[top_str] = {type(parameters[top_str])}")
        if isinstance(parameters[top_str], (tuple, np.ndarray, float, int, list)):
            middle_str_arr_0.append(parameters[top_str])
            continue
        for j in range(len(number_array[1][i])):
            middle = list(parameters[top[number_array[0][i]]].keys())
            middle_str = middle[number_array[1][i][j]]
            middle_str_arr_1.append(middle_str)
            bottom_str_arr_2 = []
            #Bottom
            if isinstance(parameters[top_str][middle_str], (tuple, np.ndarray, float, int, list)):
                #Convert tuple into array
                if isinstance(parameters[top_str][middle_str], (tuple)):
                    bottom_str_arr_2.append(np.array(parameters[top_str][middle_str]))
                else:
                    bottom_str_arr_2.append(parameters[top_str][middle_str])
                #print(f"bottom_str_arr_2 = {bottom_str_arr_2}")
                bottom_str_arr_1.append(bottom_str_arr_2)
                continue
            bottom = list(parameters[top[number_array[0][i]]][middle[number_array[1][i][j]]].keys())
            #print(f"length of number array = {len(number_array)}")
            if len(number_array) > 2:
                for k in range(len(number_array[2][i][j])):
                    bottom_str_arr_2.append(bottom[int(number_array[2][i][j][k])])
            else:
                for k in range(len(bottom)):
                    bottom_str_arr_2.append(bottom[k])
            bottom_str_arr_1.append(bottom_str_arr_2)
        #print(f"bottom_str_arr_1 = {bottom_str_arr_1}")
        middle_str_arr_0.append(middle_str_arr_1)
        bottom_str_arr_0.append(bottom_str_arr_1)
        #print(f"bottom_str_arr_0 = {bottom_str_arr_0}")
    #print(f"[top_str_arr, middle_str_arr_0, bottom_str_arr_0] = {[top_str_arr, middle_str_arr_0, bottom_str_arr_0]}")
    return [top_str_arr, middle_str_arr_0, bottom_str_arr_0]

#Function to change multiple parameters
def multi_param_change(model, parameter_str_list, new_parameter_values):
    #parameter_str_list[2][:] should be same shape as new_parameter_values[:]
    #print(f"new_parameter_values = {new_parameter_values}")
    #print(f"parameter_str_list = {parameter_str_list}")
    parameters = model.get_parameters()
    for i in range(len(parameter_str_list[0])):
        #If test to check if top-key is an array or tuple 
        #Example parameters["C3"] = [-0.1648, -0.59829998, -0.63330001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if isinstance(parameters[parameter_str_list[0][i]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameters[parameter_str_list[0][i]], "keys"):
            #Replace the tuple/array with new tuple/array
            #print(f"parameters[parameter_str_list[0][i]] Before = {parameters[parameter_str_list[0][i]]}")
            if not isinstance(new_parameter_values[i], (tuple, np.ndarray, list)) and isinstance(parameters[parameter_str_list[0][i]], (tuple, np.ndarray, list)):
                parameters[parameter_str_list[0][i]] = new_parameter_values 
                continue
            parameters[parameter_str_list[0][i]] = new_parameter_values[i] 
            #new_parameter_values sent in has to be tuple/array
            #print(f"TEST")
            #print(f"parameters[parameter_str_list[0][i]] After = {parameters[parameter_str_list[0][i]]}")
            continue
        for j in range(len(parameter_str_list[1][i])):
            #if test to check if middle-key is and array or tuple
            #Example parameters["emissivities"]["feature"] = (1.0, 1.0, 1.659892404064974, 1.0675116768340536, 1.0608768682182081, 1.0, 0.8726636137878518, 1.0985346556794289, 1.1515825707787077, 0.8576380099421744)
            #print(f"parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]] = {parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]]}")
            if isinstance(parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]], (tuple, np.ndarray, int, float, list))  and not hasattr(parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]], "keys"):
                #print(f"parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]] = {parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]]}")
                #print(f"new_parameter_values = {new_parameter_values}")
                #print(f"len(new_parameter_values) = {len(new_parameter_values)}")
                if len(new_parameter_values) > 1 and isinstance(parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]], (tuple, np.ndarray, list)):
                    parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]] = new_parameter_values 
                    continue

                parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]] = new_parameter_values[i][j] 
                #print(f"parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]] = {parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]]}")
                #assume new_parameter_values[i][j] sent in has to be tuple/array
                #print(f"Test")
                continue
            for k in range(len(parameter_str_list[2][i][j])):
                parameters[parameter_str_list[0][i]][parameter_str_list[1][i][j]][parameter_str_list[2][i][j][k]] = new_parameter_values[i][j][k]
       
    model.update_parameters(parameters)
    #print(f"Multi-parameter change succesfull")

#Function to change a single parameter
def single_param_change(model, parameter_str_list, new_parameter_value):
    parameter = model.get_parameters()
    #Flatten array in case sent in as multidmensional
    parameter_str_list = flatten_list(parameter_str_list)
    #Change singular parameter
    if isinstance(parameter[parameter_str_list[0]], (tuple, np.ndarray, int, float)) and not hasattr(parameter[parameter_str_list[0]], "keys"):
        #Replace the tuple/array with new tuple/array
        #print(f"parameter[parameter_str_list[0]] = {parameter[parameter_str_list[0]]}")
        parameter[parameter_str_list[0]] = new_parameter_value 
        #print(f"new_parameter_value = {new_parameter_value}")
        #new_parameter_values sent in has to be tuple/array
    elif isinstance(parameter[parameter_str_list[0]][parameter_str_list[1]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameter[parameter_str_list[0]][parameter_str_list[1]], "keys"):
        parameter[parameter_str_list[0]][parameter_str_list[1]] = new_parameter_value
    else:
        parameter[parameter_str_list[0]][parameter_str_list[1]][parameter_str_list[2]] = new_parameter_value
    model.update_parameters(parameter)
    #print(f"Single parameter change succesfull")

#Function to flatten an inhomogenus list/array
def flatten_list(arr):
    flattened = []

    if isinstance(arr, (np.ndarray, jnp.ndarray)):
        dtype = arr.dtype
        #Converting jax-array into numpy arr so it's mutable
        #print(f"turned {arr} into numpy array")
        arr = np.array(arr)
    else:
        # Default to object if the input is not an array
        dtype = object

    for item in arr:
        if isinstance(item, (list, np.ndarray)):  # If item is a list or array
            flattened.extend(flatten_list(item))  # Flatten recursively
        else:
            flattened.append(item)  # Add non-list items

    if dtype == jnp.ndarray:#isinstance(arr, jnp.ndarray):
        return jnp.array(flattened, dtype=dtype)
    else:
        return np.array(flattened, dtype=dtype)

#Function to flatten a list/array by 1 level        
def flatten_one_level(nested_list):
    if isinstance(nested_list, (list, np.ndarray)):
        return [item for sublist in nested_list for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
    raise TypeError("Input must be a list or array.")
#Function to restructure a flattened array back into an inhomogenous 
def restructure_flattend_list(flattened, original_structure):
    #Testing
    #print(f"original_structure = {original_structure}")
    original_structure = np.array(original_structure, dtype=object) #Making sure this is an np.array

    if isinstance(flattened, (np.ndarray, jnp.ndarray)):
        dtype = flattened.dtype
        #Converting jax-array into numpy arr so it's mutable
        #print(f"turned {arr} into numpy array")
        flattened = np.array(flattened)
    else:
        # Default to object if the input is not an array
        dtype = object
    def recursive_restructure(flat_iter, structure):
        restructured = []
        for item in structure:
            if isinstance(item, (list, np.ndarray)):  # If the item is a list, recurse
                restructured.append(recursive_restructure(flat_iter, item))
            else:  # If the item is not a list, take the next value from the iterator
                restructured.append(next(flat_iter))
        #print(f"restructured = {restructured}")

        if dtype == jnp.ndarray:#isinstance(arr, jnp.ndarray):
            return jnp.array(restructured, dtype=dtype)
        elif dtype == np.ndarray:
            return np.array(restructured, dtype=dtype)
        else:
            return restructured
    flat_iter = iter(flattened)

    return recursive_restructure(flat_iter, original_structure)
#Function to get the emission
def get_emission(parameter_values_arr, parameter_nmbr_list, max, coords, downsample_step):
    #print(f"Test get emission")
    # parameter_values_arr : have the values of the parameters you want changed

    # parameter_nmbr_list: has the numbers that correspond to the parameters you want changed. Neccessary because Zodipy doesn't allow sending in strings

    #Max: maximum size of the emission model

    model = zodipy.Model(30 * u.micron)
    #parameters = model.get_parameters()
    #Need to convert nmbr into str-list 
    parameter_str_arr = parameter_nmbr_to_str_list(parameter_nmbr_list, model) 
    #print(f"TEST")
    #print(f"parameter_str_arr = {parameter_str_arr}")
    #print(f"parameter_values_arr = {parameter_values_arr}")
    if isinstance(parameter_values_arr, (int, float, str, bool)):
        single_param_change(model, parameter_str_arr, parameter_values_arr)
    else:
        multi_param_change(model, parameter_str_arr, parameter_values_arr)

    #Skip every 10th coord to downsample. Add downsampling argument()
    #print(f"TEST")
    emission = model.evaluate(coords[0:max:downsample_step], nprocesses=1) #size of 68828. Can decide the size using [0:1000:10]
    #print(f"emission = {emission}")
    #print(f"len(emission) = {len(emission)}")
    return emission

#Function to get the gradient of S_zodi
def get_zodigrad(parameter_values_arr, parameter_nmbr_list, max, coords, downsample_step):
    dS_zodidOmega = jax.jacfwd(get_emission)(parameter_values_arr, parameter_nmbr_list, max, coords, downsample_step)
    return dS_zodidOmega

#Getting CHI squared 
def get_chi2(parameter_values_arr, parameter_nmbr_list, max, coords, noisy_data, downsample_step):
    #print(f"Test Chi2")
    chi2 = 0
    sigma = 0.1
    emission = get_emission(parameter_values_arr, parameter_nmbr_list, max, coords, downsample_step)
    if type(emission) == jax.interpreters.ad.JVPTracer:
        chi2 = jnp.sum(((noisy_data.value - emission)/sigma)**2)
    else:
        chi2 = jnp.sum(((noisy_data.value - emission.value)/sigma)**2)
    return chi2

#Function to get the gradient of chi2
def get_chi2_gradient(parameter_values_arr, parameter_nmbr_list, max, coords, noisy_data, downsample_step):
    dchi2_dOmega = jnp.array(jax.jacfwd(get_chi2)(parameter_values_arr, parameter_nmbr_list, max, coords, noisy_data, downsample_step))
    #print(f"dchi2_dOmega = {dchi2_dOmega}")
    return dchi2_dOmega

#Function to make the noise
def noise_maker(data, mu, sigma): 
    noisy_data = jnp.zeros(len(data))*data.unit
    for i in range(len(data)):
        n = np.random.normal(mu, sigma)*data.unit
        noisy_data[i] = data[i] + n
    return noisy_data

#Function to apply HMC
def HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma_array, model, coords, downsample_step):
    max = 68828 #Length of the emission dataset. 
    #Because parameterVal_paramNr[0] is potentially inhomogenous we just deep copy and change the values as we go
    #p = np.zeros_like(parameterVal_paramNr[0]) #Doesn't work with inhomogenous
    p = copy.deepcopy(parameterVal_paramNr[0]) #Works with inhomogenous and homogenous
    mu = 0
    #"""
    #Testing
    #Flatten arrays to make them easier to work with
    #print(f"p = {p}")
    flat_p = flatten_list(p)
    flat_parameterVal_paramNr_0 = flatten_list(parameterVal_paramNr[0]) 
    flat_sigma_array = flatten_list(sigma_array)
    for i in range(len(flat_parameterVal_paramNr_0)):
        flat_p[i] = np.random.normal(mu, flat_sigma_array[i])
    
    flat_p = jnp.array(flat_p) #Make it into an jnp-array
    parameter_arr = copy.deepcopy(parameterVal_paramNr)
    parameter_arr[1][:] = copy.deepcopy(parameterVal_paramNr[1][:])
    parameter = jnp.array(parameterVal_paramNr[0][:])
    flat_parameter = jnp.array(flat_parameterVal_paramNr_0) 
    current_p = flat_p[:]
    #print(f"Test 1")
    #print(f"parameter = {parameter}")
    #print(f"parameterVal_paramNr[1][:] = {parameterVal_paramNr[1][:]}")
    Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step) 
    #need to flatten Chi2_grad
    #print(f"Test 2")
    flat_Chi2_grad = flatten_list(Chi2_grad)
    flat_p = flat_p - epsilon*flat_Chi2_grad/2
    for j in range(1, steps, steps-1):
        flat_parameter = flat_parameter + epsilon*flat_p
        if (j!=steps):
            #Convert back into non-flat array
            #print(f"flat_parameter = {flat_parameter}")
            #print(f"original parameter structure = {parameter}")
            parameter = restructure_flattend_list(flat_parameter, parameter)
            #print(f"new parameter structure = {parameter}")
            #print(f"flat_p = {flat_p}")
            #print(f"flat_Chi2_grad = {flat_Chi2_grad}")
            Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
            flat_Chi2_grad = flatten_list(Chi2_grad)
            flat_p = flat_p - epsilon * flat_Chi2_grad 
    Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
    flat_Chi2_grad = flatten_list(Chi2_grad)
    flat_p = flat_p - epsilon*flat_Chi2_grad/2
    flat_p = -flat_p 
    #print(f"parameterVal_paramNr[1][:] = {parameterVal_paramNr[1][:]}")
    #print(f"Current param = {parameterVal_paramNr[0][:]}")
    current_Chi2 = get_chi2(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)
    current_K = np.sum(current_p**2)/2
    #Restructure parameter again
    parameter = restructure_flattend_list(flat_parameter, parameter)
    #print(f"Proposed param = {parameter}")
    proposed_Chi2 = get_chi2(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
    proposed_K = np.sum(flat_p**2)/2

    rng = np.random.default_rng()
    number = rng.random()
    probability = current_Chi2-proposed_Chi2+current_K-proposed_K
    #print(f"parameter = {parameter}")
    #print(f"current_Chi2 = {current_Chi2}")
    #print(f"proposed_Chi2 = {proposed_Chi2}")
    #print(f"Probability = {probability}")
    #print(f"number = {number}")
    """               
    for i in range(len(parameterVal_paramNr[0])):
        for j in range(len(parameterVal_paramNr[0][i])):
            for k in range(len(parameterVal_paramNr[0][i][j])):
                p[i][j][k] = np.random.normal(mu, sigma_array[i][j][k])
    parameter_arr = parameterVal_paramNr
    parameter_arr[1][:] = parameterVal_paramNr[1][:]
    parameter = jnp.array(parameterVal_paramNr[0][:]) 
    current_p = p[:]
    Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
    p = p - epsilon*Chi2_grad/2
    for j in range(1, steps, steps-1):
        parameter = parameter + epsilon*p
        if (j!=steps):
            Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
            p = p - epsilon * Chi2_grad 

    Chi2_grad = get_chi2_gradient(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
    p = p - epsilon*Chi2_grad/2

    p = -p 

    current_Chi2 = get_chi2(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)
    current_K = np.sum(current_p**2)/2
    proposed_Chi2 = get_chi2(parameter, parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)  
    proposed_K = np.sum(p**2)/2

    rng = np.random.default_rng()
    number = rng.random()
    probability = current_Chi2-proposed_Chi2+current_K-proposed_K
    #"""
    if number < np.exp(probability):
        parameter_arr[0][:] = np.array(parameter, dtype='float64')
        #Also return the emission
        proposed_emission = get_emission(parameter, parameterVal_paramNr[1][:], max, coords, downsample_step)
        return parameter_arr, 1, proposed_Chi2, proposed_emission #accepted
    else:
        #Also return the emission
        current_emission = get_emission(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, downsample_step)
        return parameterVal_paramNr, 0,  current_Chi2, current_emission #rejected

#Code that tunes the stepsize
def tuning_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma, model, downsample_step, iterations, coords):
    print(f"\nTuning")
    #Tuning to find good fitting epsilon
    accepted = 0
    #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
    #print(f"First {parameterVal_paramNr[0]}")
    #print(f"Second {new_parameterVal_paramNr[0]}")
    iterations_tuning = iterations//10
    if iterations_tuning < 10:
        iterations_tuning = 10
    epsilon_arr = []
    AP_arr = []
    iteration_value_arr = []
    #print(f"iterations = {iterations}")
    end_tuning = False
    close_tuning = False
    while iterations_tuning < iterations:
        new_parameterVal_paramNr = copy.deepcopy(parameterVal_paramNr)
        if iterations_tuning == iterations - 1:
            iterations_tuning = iterations
        iteration_value_arr = np.append(iteration_value_arr, iterations_tuning) 
        epsilon_arr = np.append(epsilon_arr, epsilon)
        #Documenting in case we find a good epsilon early
        chi2 = get_chi2(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)
        emission = get_emission(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, downsample_step)
        parameter_value_arr = np.array([parameterVal_paramNr[0][:]], dtype='float64')
        chi2_array = np.array(chi2, dtype="float64")
        emission_array = np.array(emission.value, dtype="float64")

        for i in tqdm(range(iterations_tuning-1)):
            new_parameterVal_paramNr, result, chi2, emission = HMC(epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma, model, coords, downsample_step)
            accepted += result
            #Documenting in case we find a good epsilon early
            parameter_value_arr = np.vstack([parameter_value_arr, np.array([new_parameterVal_paramNr[0]])])
            chi2_array = np.append(chi2_array, chi2)  
            emission_array = np.append(emission_array, emission.value)
        acceptance_probability = accepted/iterations_tuning
        print(f"acceptance probability = {acceptance_probability}")
        AP_arr = np.append(AP_arr, acceptance_probability)
        #0.4-0.9 are good targets
        lower_bound = 0.4
        upper_bound = 0.9

        #Debugging values
        #lower_bound = 0.01
        #upper_bound = 0.99
        while acceptance_probability < lower_bound or acceptance_probability > upper_bound:
            if acceptance_probability == 0 and close_tuning == False:
                epsilon = epsilon*0.5
                print(f"Reduce by 0.5")
            elif acceptance_probability < lower_bound*0.9 and close_tuning == True:
                epsilon = epsilon*0.9
                print(f"Reduce by 0.9")
            elif acceptance_probability < lower_bound:
                epsilon = epsilon*0.999 #Reduce epsilon by 1%
                close_tuning = True
                print(f"Reduce by 0.999")
            if acceptance_probability == 1 and close_tuning == False:
                epsilon = epsilon*5 
                print(f"Increase by 5")
            elif acceptance_probability > upper_bound*1.1 and close_tuning == True:
                epsilon = epsilon*1.1
                print(f"Increase by 1.1")
            elif acceptance_probability > upper_bound:
                epsilon = epsilon*1.001 #Increase epsilon by 10%
                close_tuning = True
                print(f"Increase by 1.001")
            #Stops the while loop when epsilon is bigger than 10
            if epsilon > 10:
                break
            print(f"Current epsilon: {epsilon}") 
            epsilon_arr = np.append(epsilon_arr, epsilon) #Put all the epsilons in a list 
            #Running again to see if the epsilon improved
            accepted = 0   
            new_parameterVal_paramNr = copy.deepcopy(parameterVal_paramNr)
            #Making the arrays and documenting, will be reset every while loop. Stop when achieved good acceptance rate
            chi2 = get_chi2(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, noisy_data, downsample_step)
            emission = get_emission(parameterVal_paramNr[0][:], parameterVal_paramNr[1][:], max, coords, downsample_step)
            parameter_value_arr = np.array([parameterVal_paramNr[0][:]], dtype='float64')
            chi2_array = np.array(chi2, dtype="float64")
            emission_array = np.array(emission.value, dtype="float64")

            for i in tqdm(range(iterations_tuning-1)):
                new_parameterVal_paramNr, result, chi2, emission = HMC(epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma, model, coords, downsample_step)
                accepted += result
                #Documenting
                parameter_value_arr = np.vstack([parameter_value_arr, np.array([new_parameterVal_paramNr[0]])])
                chi2_array = np.append(chi2_array, chi2)   
                emission_array = np.append(emission_array, emission.value)            
            acceptance_probability = accepted/iterations_tuning
            AP_arr = np.append(AP_arr, acceptance_probability)
            iteration_value_arr = np.append(iteration_value_arr, iterations_tuning) 
            print(f"acceptance probability = {acceptance_probability}")
        accepted = 0
        iterations_tuning *= 10
        if end_tuning:
            #print(f"iterations_tuning = {iterations_tuning}")
            print(f"Ending while loop")
            break
        if iterations_tuning >= iterations:
            iterations_tuning = iterations - 1
            end_tuning = True
        #print(f"iterations_tuning = {iterations_tuning}")
        

    #Making txt files of epsilon and the acceptance rate
    with open(f"Images_3/HMC_epsilon_Acc_rate_{np.array(new_parameterVal_paramNr[0]).size}_parameters.txt", 'w') as file:
            file.write(f"Epsilon\tAcceptance Rate\t Iterations\n")
            for i in range(len(epsilon_arr)):
                file.write(f"{epsilon_arr[i]} \t {AP_arr[i]} \t {iteration_value_arr[i]}\n")
    print(f"Epsilons written to files")
    print(f"End of tuning \n")
    print(f"Best epilson = {epsilon}")
    #print(f"Epsilon_arr[-1] = {epsilon_arr[-1]}")
    return epsilon, parameter_value_arr, chi2_array, acceptance_probability, emission_array

#Function to send in the default values for 
def get_default_parameter_values(parameterVal_paramNr, param_strings, model):
    parameters = model.get_parameters()
    #print(f"parameterVal_paramNr in get default = {parameterVal_paramNr}")
    default_parameterVal = copy.deepcopy(parameterVal_paramNr[1][2])
    #print(f"default_parameterVal = {default_parameterVal}")
    for i in range(len(default_parameterVal)):
        if isinstance(parameters[param_strings[0][i]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameters[param_strings[0][i]], "keys"):
            default_parameterVal[i] = parameters[param_strings[0][i]]
            continue
        for j in range(len(default_parameterVal[i])):
            if isinstance(parameters[param_strings[0][i]][param_strings[1][i][j]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameters[param_strings[0][i]][param_strings[1][i][j]], "keys"):
                default_parameterVal[i][j] = parameters[param_strings[0][i]][param_strings[1][i][j]]
                continue
            for k in range(len(default_parameterVal[i][j])):
                default_parameterVal[i][j][k] = parameters[param_strings[0][i]][param_strings[1][i][j]][param_strings[2][i][j][k]]             
    #print(f"default_parameterVal = {default_parameterVal}")       
    return np.array(default_parameterVal, dtype="float64")

#Function to find the sigma
def find_sigma(parameterVal_paramNr, noisy_data, filename, coords, model, downsample_step):
    print(f"Finding Optimal Sigma")
    #Convert the strings in parameterVal_paramNr into numbers for JAX in case they are sent in as strings
    if any(isinstance(x, str) for sublist in parameterVal_paramNr[1] for x in np.ravel(sublist)):
        #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
        param_strings = parameterVal_paramNr[1]
        parameterVal_paramNr[1] = parameter_list_str_to_nmbr(parameterVal_paramNr[1], model)
        #print(f"Updated parameterVal_paramNr = {parameterVal_paramNr}")

    #Put the parameter strings into an array for later use
    if any(isinstance(x, (int, float, np.integer, np.floating)) for sublist in parameterVal_paramNr[1] for x in np.ravel(sublist)):
        param_strings = parameter_nmbr_to_str_list(parameterVal_paramNr[1] , model)
        #print(f"param_strings = {param_strings}")
    #Code snippet that returns the already established values if we send in an empty array of parameter values
    if np.shape(parameterVal_paramNr[0]) != np.shape(parameterVal_paramNr[1][2]) or len(flatten_list(parameterVal_paramNr[0])) < 1:
        print(f"Parameter array that was sent in is a different shape from parameter number array")
        print(f"Using default values for the given parameter number array")
        """
        parameters = model.get_parameters()
        default_parameterVal = np.zeros_like(parameterVal_paramNr[1][2], dtype='float64')
        for i in range(len(parameterVal_paramNr[1][0])):
            for j in range(len(parameterVal_paramNr[1][1][i])):
                for k in range(len(parameterVal_paramNr[1][2][i][j])):
                    default_parameterVal[i][j][k] = parameters[param_strings[0][i]][param_strings[1][i][j]][param_strings[2][i][j][k]] 
        #Change into default values
        #print(f"default_parameterVal = {default_parameterVal}")            
        parameterVal_paramNr[0] = default_parameterVal
        """
        parameterVal_paramNr[0] = copy.deepcopy(get_default_parameter_values(parameterVal_paramNr, param_strings, model))
    max = 68828 #Length of the emission dataset. 
    limit = 1 #Chi2 difference change
    flat_param_val = flatten_list(copy.deepcopy(parameterVal_paramNr[0]))
    #print(f"flat_param_val = {flat_param_val}")
    flat_change = np.ones_like(flat_param_val)
    #print(f"flat change = {flat_change}")
    print(f"parameterVal_paramNr[0] = {parameterVal_paramNr[0]}")
    change = restructure_flattend_list(flat_change, parameterVal_paramNr[0])
    parameter = model.get_parameters()
    #print(f"change = {change}")
    #print(f"parameterVal_paramNr = {parameterVal_paramNr}")

    #if 
    for i in tqdm(range(len(change))):
        #For when top keys have values
        if isinstance(parameter[param_strings[0][i]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameter[param_strings[0][i]], "keys"):
            new_array = [[parameterVal_paramNr[1][0][i]]]
            #print(f"new_array = {new_array}")
            #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
            difference = 1000000
            #Check if it's a tuple, array or list and apply a specialised way to find change
            if isinstance(parameter[param_strings[0][i]], (tuple, np.ndarray, list)):
                parameter_array = parameterVal_paramNr[0][i]
                parameter_1 = parameterVal_paramNr[0][i]
                Chi2_1 = get_chi2(parameter_1, new_array, max, coords, noisy_data, downsample_step)
                for m in range(len(change[i])):
                    difference = 1000000
                    while difference > limit:
                        change[i][m] *= 0.1
                        parameter_2 = parameterVal_paramNr[0][i][m] + change[i][m]
                        parameter_array[m] = parameter_2
                        Chi2_2 = get_chi2(parameter_array, new_array, max, coords, noisy_data, downsample_step)
                        difference = np.mean(abs(Chi2_1 - Chi2_2))
                continue
            parameter_1 = parameterVal_paramNr[0][i]
            Chi2_1 = get_chi2(parameter_1, new_array, max, coords, noisy_data, downsample_step)
            while difference > limit:
                change[i] *= 0.1
                parameter_2 = parameterVal_paramNr[0][i] + change[i]
                Chi2_2 = get_chi2(parameter_2, new_array, max, coords, noisy_data, downsample_step)
                difference = np.mean(abs(Chi2_1 - Chi2_2))
            continue
        for j in range(len(change[i])):
            #For when middle keys have values
            if isinstance(parameter[param_strings[0][i]][param_strings[1][i][j]], (tuple, np.ndarray, int, float, list)) and not hasattr(parameter[param_strings[0][i]][param_strings[1][i][j]], "keys"):
                new_array = [[parameterVal_paramNr[1][0][i]], [[parameterVal_paramNr[1][1][i][j]]]]
                difference = 1000000
                #Check if it's a tuple, array or list and apply a specialised way to find change
                if isinstance(parameter[param_strings[0][i]][param_strings[1][i][j]], (tuple, np.ndarray)):
                    parameter_array = parameterVal_paramNr[0][i][j]
                    parameter_1 = parameterVal_paramNr[0][i][j]
                    #print(f"new_array = {new_array}")
                    #print(f"parameter_1 = {parameter_1}")
                    Chi2_1 = get_chi2(parameter_1, new_array, max, coords, noisy_data, downsample_step)
                    for m in range(len(change[i])):
                        difference = 1000000
                        while difference > limit:
                            change[i][j][m] *= 0.1
                            parameter_2 = parameterVal_paramNr[0][i][j][m] + change[i][j][m]
                            parameter_array[m] = parameter_2
                            Chi2_2 = get_chi2(parameter_array, new_array, max, coords, noisy_data, downsample_step)
                            difference = np.mean(abs(Chi2_1 - Chi2_2))
                    continue           
                parameter_1 = parameterVal_paramNr[0][i][j]
                Chi2_1 = get_chi2(parameter_1, new_array, max, coords, noisy_data, downsample_step)
                while difference > limit:
                    change[i][j] *= 0.1
                    parameter_2 = parameterVal_paramNr[0][i][j] + change[i][j]
                    Chi2_2 = get_chi2(parameter_2, new_array, max, coords, noisy_data, downsample_step)
                    difference = np.mean(abs(Chi2_1 - Chi2_2))
                continue
            for k in range(len(change[i][j])):
                #For when bottom keys have values
                new_array = [[parameterVal_paramNr[1][0][i]], [[parameterVal_paramNr[1][1][i][j]]], [[[int(parameterVal_paramNr[1][2][i][j][k])]]]]
                difference = 1000000
                parameter_1 = parameterVal_paramNr[0][i][j][k]
                Chi2_1 = get_chi2(parameter_1, new_array, max, coords, noisy_data, downsample_step)
                while difference > limit:
                    change[i][j][k] *= 0.1 
                    parameter_2 = parameterVal_paramNr[0][i][j][k] + change[i][j][k]
                    Chi2_2 = get_chi2(parameter_2, new_array, max, coords, noisy_data, downsample_step)
                    difference = np.mean(abs(Chi2_1 - Chi2_2))
    with open(f"Images_3/{filename}_sigma_{len(np.ravel(parameterVal_paramNr[1][2]))}parameters.txt", "w") as file:
        formatted_data = f"{change}".replace(" ", ", ")
        file.write(f"{formatted_data}")
    file.close()
    print(f"Saved optimal sigma into a txt-file\n")
    print(f"sigma = {change}")
    return change

#Function applies HMC and returns an array of the parameters
def get_parameter_HMC(epsilon, parameterVal_paramNr, noisy_data, model, coords, iterations, sigma, filename, downsample_step, steps):
    print(f"Getting parameter values from HMC")
    #Steps = 10
    #Convert the strings in parameterVal_paramNr into numbers for JAX in case they are sent in as strings
    if any(isinstance(x, str) for sublist in parameterVal_paramNr[1] for x in np.ravel(sublist)):
        #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
        param_strings = parameterVal_paramNr[1]
        parameterVal_paramNr[1] = parameter_list_str_to_nmbr(parameterVal_paramNr[1], model)
        print(f"Updated parameterVal_paramNr = {parameterVal_paramNr}")

    #Put the parameter strings into an array for later use
    if any(isinstance(x, (int, float, np.integer, np.floating)) for sublist in parameterVal_paramNr[1] for x in np.ravel(sublist)):
        param_strings = parameter_nmbr_to_str_list(parameterVal_paramNr[1] , model)
        
    #Code snippet that returns the already established values if we send in an empty array of parameter values
    if  np.shape(parameterVal_paramNr[0]) != np.shape(parameterVal_paramNr[1][2]) or len(flatten_list(parameterVal_paramNr[0])) < 1:
        print(f"Parameter array that was sent in is a different shape from parameter number array")
        print(f"Using default values for the given parameter number array")
        """
        parameters = model.get_parameters()
        default_parameterVal = np.zeros_like(parameterVal_paramNr[1][2], dtype='float64')
        for i in range(len(parameterVal_paramNr[1][0])):
            for j in range(len(parameterVal_paramNr[1][1][i])):
                for k in range(len(parameterVal_paramNr[1][2][i][j])):
                    default_parameterVal[i][j][k] = parameters[param_strings[0][i]][param_strings[1][i][j]][param_strings[2][i][j][k]] 
        #Change into default values
        #print(f"default_parameterVal = {default_parameterVal}")            
        parameterVal_paramNr[0] = default_parameterVal
        """
        parameterVal_paramNr[0] = copy.deepcopy(get_default_parameter_values(parameterVal_paramNr, param_strings, model))
    #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
    good_fit_epsilon, parameter_value_arr, chi2_array, acceptance_probability, emission_array = tuning_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma, model, downsample_step, iterations, coords)
    #print(f"good_fit_epsilon = {good_fit_epsilon}")
    #good_fit_epsilon = 1.8433e-05 #Using this for bug testing
    #good_fit_epsilon = 0.039653083735466284 #From file
    #good_fit_epsilon = 0.03776484165282503
    """
    accepted = 0
    #Initial parameter values
    new_parameterVal_paramNr, result, chi2 = HMC(good_fit_epsilon, steps, parameterVal_paramNr, noisy_data, sigma, model, coords, downsample_step)
    accepted += result
    parameter_value_arr = np.array([new_parameterVal_paramNr[0][:]], dtype='float64')
    chi2_array = np.array(chi2, dtype="float64")
    for i in tqdm(range(iterations-1)):
        new_parameterVal_paramNr, result, chi2 = HMC(good_fit_epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma, model, coords, downsample_step)
        accepted += result
        parameter_value_arr = np.vstack([parameter_value_arr, np.array([new_parameterVal_paramNr[0]])])
        chi2_array = np.append(chi2_array, chi2)
    #"""
    print(f"Acceptance rate: {acceptance_probability}")

    #Write to a file
    with open(f"Images_3/{filename}_accept_rate.txt", "w") as file:
        file.write(f"Epsilon: {good_fit_epsilon}")
        file.write(f"Acceptance rate: {acceptance_probability}")
    file.close()
    parameters = model.get_parameters()
    #FIX ME. I DON'T WORK WITH THE WEORD ASS TUPLES/FLOATS/INTS
    with open(f"Images_3/{filename}.txt", 'w') as file:
            for i in range(len(param_strings[0])):
                #Check if top keys have int, float, array or tuple
                if not hasattr(parameters[param_strings[0][i]], "keys"):
                    if isinstance(parameters[param_strings[0][i]], (tuple, np.ndarray, list)):
                        for m in range(len(parameters[param_strings[0][i]])):
                            file.write(f"{param_strings[0][i]}_{m}\t")
                    elif isinstance(parameters[param_strings[0][i]], (int, float)):
                        file.write(f"{param_strings[0][i]}\t")
                    continue
                for j in range(len(param_strings[1][i])):

                    #Check if middle keys have int, float, array of tuple
                    if not hasattr(parameters[param_strings[0][i]][param_strings[1][i][j]], "keys"):
                        if isinstance(parameters[param_strings[0][i]][param_strings[1][i][j]], (tuple, np.ndarray, list)):
                            for m in range(len(parameters[param_strings[0][i]][param_strings[1][i][j]])):
                                file.write(f"{parameters[param_strings[0][i]][param_strings[1][i][j]]}_{m}\t")
                        elif isinstance(parameters[param_strings[0][i]][param_strings[1][i][j]], (int, float)):
                            file.write(f"{parameters[param_strings[0][i]][param_strings[1][i][j]]}\t")
                        continue
                    for k in range(len(param_strings[2][i][j])):
                        file.write(f"{param_strings[2][i][j][k]}\t")
            file.write(f"chi2\t")
            file.write(f"\n")
            for i in range(iterations):
                for j in range(len(parameter_value_arr[i])):

                    #Check if top keys have int, float, array or tuple
                    if not hasattr(parameters[param_strings[0][j]], "keys"):
                        if isinstance(parameters[param_strings[0][j]], (tuple, np.ndarray, list)):
                            for m in range(len(parameters[param_strings[0][j]])):
                                file.write(f"{parameter_value_arr[i][j][m]}\t")
                        elif isinstance(parameters[param_strings[0][j]], (int, float)):
                            file.write(f"{parameter_value_arr[i][j]}\t")
                        continue

                    for k in range(len(parameter_value_arr[i][j])):

                        #Check if middle keys have int, float, array of tuple
                        if not hasattr(parameters[param_strings[0][j]][param_strings[1][j][k]], "keys"):
                            if isinstance(parameters[param_strings[0][j]][param_strings[1][j][k]], (tuple, np.ndarray, list)):
                                for m in range(len(parameters[param_strings[0][j]][param_strings[1][j][k]])):
                                    file.write(f"{parameter_value_arr[i][j][k][m]}\t")
                            elif isinstance(parameters[param_strings[0][j]][param_strings[1][j][k]], (int, float)):
                                file.write(f"{parameter_value_arr[i][j][k]}\t")
                            continue

                        for l in range(len(parameter_value_arr[i][j][k])):
                            file.write(f"{parameter_value_arr[i][j][k][l]}\t")
                file.write(f"{chi2_array[i]}\t")
                file.write(f"\n")
    file.close()
    print(f"Values written to the file Images_3/{filename}.txt")
    #Writing the Emission values 
    with open(f"Images_3/{filename}_Emission.txt", "w") as file:
        for i in range(len(emission_array)):
            file.write(f"{emission_array[i]}")
    file.close()
    return parameter_value_arr, param_strings, chi2_array, emission_array

#Function to make a corner plot
def get_corner_plt(parameter_value_array, Chi2_array, parameter_name, iterations):
    #Removing large Chi2 values 
    threshold = np.mean(Chi2_array)
    filter = Chi2_array <= threshold
    Chi2_array = Chi2_array[filter]
    parameter_value_array = parameter_value_array[filter, :]
    #print(f"parameter value array = {parameter_value_array}")
    #print(f"Chi2 array = {Chi2_array}")
    #Combining parameter and Chi2
    combo = np.hstack([parameter_value_array, Chi2_array.reshape(-1, 1)])
    result = np.vstack(combo)
    #print(f"shape of resut: {np.shape(result)}")
    parameter_name = np.append(parameter_name, ["Chi2"], axis=0)
    labels = parameter_name
    figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.savefig(f"Images_6/HMC_corner_{iterations}_iterations_{len(parameter_name)}_parameters.png")
    plt.close()
    #Plotting Chi2 against iterations
    plt.plot(Chi2_array)
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.savefig(f"Images_6/HMC_corner_{iterations}_iterations_Chi2.png")
    plt.close()

#Corner plot using txt files
def get_corner_plt_readfile(filename, file_filename, truth_dict):
    with open(filename, 'r') as file:
        labels = file.readline().strip().split()
    data = np.loadtxt(filename, skiprows = 1)
    iterations = len(data[:,-1])

    #Plotting Chi2 against iterations unfiltered
    plt.plot(data[:,-1])
    plt.title(f"Chi2 unfiltered")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.savefig(f"Images_3/{file_filename}_Chi2_unfiltered.png")
    plt.close()

    #Burn-in removal 
    #threshold = np.mean(data[:,-1])
    #filter = data[:, -1] < threshold
    burn_in = 1500 #Burn in 10%-50% (???) take middle ground of 30%
    data = data[burn_in:]
    mean_val = np.mean(data, axis=0)
    mean_paramters = np.mean(data[:-1,:-1], axis=0)
    result = np.vstack(data)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict, labels = labels)
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.savefig(f"Images_3/{file_filename}_{len(data[:][0])-1}_parameters.png")
    plt.close()
    #Plotting Chi2 against iterations
    plt.plot(data[:,-1])
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.yscale("log")
    plt.savefig(f"Images_3/{file_filename}_Chi2.png")
    plt.close()
    return mean_val, mean_paramters
"""
'x_0': 0.011887800744346281, 
'y_0': 0.005476506466226378, 
'z_0': -0.0021530908020710744, 
'i': 2.033518807239077, 
'Omega': 77.65795555409711, 
'n_0': 1.134437388142796e-07, 
'alpha': 1.337069670593028, 
'beta': 4.141500415758664, 
'gamma': 0.9420617939335804, 
'mu': 0.1887317648909019
"""
"""
Omega = 77.658  #77.65795555409711 
Omega_idx = 4
gamma = 0.942 #0.9420617939335804
gamma_idx = 8 
beta = 4.14#4.141500415758664
beta_idx = 7
alpha = 1.34#1.337069670593028
alpha_idx = 6

#parameters = [Omega, gamma, alpha, beta]
x_0 =  0.012
y_0 = 0.005
z_0 = -0.002 
i= 2.033 
Omega = 77.657 
n_0 = 1.134
alpha = 1.337 
beta = 4.142 
gamma = 0.942 
mu = 0.189

#parameters = [[[x_0, y_0, z_0, i, Omega, n_0, alpha, beta, gamma, mu]]]
#Testing out the code
#epsilon = 3.2e-06
epsilon = 1.2211162399298152e-05
#1.2211162399298152e-05 give 0.25
steps = 10 #IMPORTANT TO HAVE 10 STEPS AS THAT'S WHAT I USED DURING TUNING
iterations = 10 #Important, need enough iterations so the values have some variance for corner.

#parameter_str = ["comps", "cloud", ["Omega", "gamma", "alpha", "beta"]] #Example of what to send in
#parameters = [x_0, y_0, z_0, i, Omega, n_0, alpha, beta, gamma, mu]
#parameters = [[[x_0, y_0, z_0, i, Omega, alpha, beta, gamma, mu]]] #Trying without n_0
parameters = [[[]]] #testing default function
parameter_str = [["comps"], [["cloud"]], [[["x_0", "y_0", "z_0", "i", "Omega", "alpha", "beta", "gamma", "mu"]]]] #Send in empty to get all the parameters
parameterVal_paramNr = [parameters, parameter_str]

parameter_values_arr = parameters

max = 68828

comm_tod = TODLoader("/mn/stornext/d23/cmbco/cg/dirbe/data", "DIRBE")
comm_tod.init_file('05_nside512_V23', '')
pix = comm_tod.load_field(f'000185/05_A/pix')
nside = 512 #hp.npix2nside(pix)
lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
t0 = Time("2022-01-14") #MJD
dt = TimeDelta(1, format="sec")
#Have t0  at different days
obstimes = t0 + jnp.arange(lat.size) * dt
coords = SkyCoord(
        lon,
        lat, 
        unit=u.deg, 
        frame="galactic", #Need to put in Galactic coordinates?
        obstime=obstimes #Need array of obstimes 
)

downsample_step = 10
#print(f"Parameter values arr = {parameter_values_arr}")
model = zodipy.Model(30 * u.micron)
parameter_nmbr_list = parameter_list_str_to_nmbr(parameter_str, model)

#Making noisy sample
parameters_OG = [[[x_0, y_0, z_0, i, Omega, alpha, beta, gamma, mu]]]
parameter_str_OG = [["comps"], [["cloud"]], [[["x_0", "y_0", "z_0", "i", "Omega", "alpha", "beta", "gamma", "mu"]]]] #Send in empty to get all the parameters
parameterVal_paramNr_OG = [parameters_OG, parameter_str_OG]
parameter_nmbr_list_OG = parameter_list_str_to_nmbr(parameter_str_OG, model)
emission_og = get_emission(parameters_OG, parameter_nmbr_list_OG, max, coords, downsample_step) #BE CAREFUL YOU NEED TO MAKE SURE YOU HAVE tHE RIGHT EMISSION
noisy_data = noise_maker(emission_og, 0, 0.1) #(emission, mu, sigma)


filename = f"HMC_cloud_param_test_{iterations}_iterations"
#sigma = find_sigma(parameterVal_paramNr, noisy_data, filename)
#print(f"sigma = {sigma}")
#sigma = [9.999999e-04, 9.999999e-09, 9.999999e-07, 9.999999e-07]
#sigma = [1.e-09, 1.e-10, 1.e-10, 1.e-08, 1.e-06, 1.e-16, 1.e-09, 1.e-09, 1.e-09, 1.e-10]
sigma = [[[1.e-07, 1.e-07, 1.e-06, 1.e-06, 1.e-05, 1.e-06, 1.e-06, 1.e-06, 1.e-07]]] #Without n_0
#parameter_value_arr, parameter_name, Chi2_array = get_parameter_HMC(epsilon, parameterVal_paramNr, noisy_data, model, coords, iterations, sigma, filename, downsample_step, steps=10)
#"""
#Overhead function 
def get_parameter_HMC_function(epsilon, parameterVal_paramNr, noisy_data, model, dates, iterations, filename, downsample_step, sigma_file, steps):
    #Fills in parameters if parameterVal_paramNr was sent in with only either the first or second keys


    #Convert 
    #Checks if there are any strings that are sent in
    if any(isinstance(x, str) for sublist in parameterVal_paramNr[1] for x in np.ravel(sublist)):
        parameterVal_paramNr[1] = parameter_list_str_to_nmbr(parameterVal_paramNr[1], model)
        print(f"Changed the strings into integers for JAX readability")

    comm_tod = TODLoader("/mn/stornext/d23/cmbco/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix')
    nside = 512 #hp.npix2nside(pix)
    lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
    #lon, lat = hp.pix2ang(nside, pix, lonlat=True) 
    #Find optimal Sigma, we assume we can use same sigma for all the days
    if sigma_file == False:
        #Find optimal sigma
        t0 = Time(dates[0]) #MJD
        dt = TimeDelta(1, format="sec")
        #Have t0  at different days
        obstimes = t0 + jnp.arange(lat.size) * dt
        coords = SkyCoord(
                lon,
                lat, 
                unit=u.deg, 
                frame="galactic", #Need to put in Galactic coordinates?
                obstime=obstimes #Need array of obstimes 
        )
        parameterVal_paramNr_copy = copy.deepcopy(parameterVal_paramNr)
        sigma = find_sigma(parameterVal_paramNr_copy, noisy_data, filename, coords, model, downsample_step)
    else:
        print(f"sigma_file = {sigma_file}")
        with open(sigma_file, 'r') as file:
            values = file.read()
            sigma = ast.literal_eval(values)
        file.close()
        sigma = np.array(sigma, dtype="float64")
        print(f"Sigma succesfully read from file")
        print(f"sigma = {sigma}")


    if len(dates) > 1:
        for i in range(len(dates)):
            t0 = Time(dates[i]) #MJD
            dt = TimeDelta(1, format="sec")
            #Have t0  at different days
            obstimes = t0 + jnp.arange(lat.size) * dt
            coords = SkyCoord(
                    lon,
                    lat, 
                    unit=u.deg, 
                    frame="galactic", #Need to put in Galactic coordinates?
                    obstime=obstimes #Need array of obstimes 
            )
            parameter_value_arr, parameter_name, Chi2_array = get_parameter_HMC(epsilon, parameterVal_paramNr, noisy_data, model, coords, iterations, sigma, filename, downsample_step, steps)
    
    else:
        t0 = Time(dates[0]) #MJD
        dt = TimeDelta(1, format="sec")
        #Have t0  at different days
        obstimes = t0 + jnp.arange(lat.size) * dt
        coords = SkyCoord(
                lon,
                lat, 
                unit=u.deg, 
                frame="galactic", #Need to put in Galactic coordinates?
                obstime=obstimes #Need array of obstimes 
        )
        parameter_value_arr, parameter_name, Chi2_array, emission_array = get_parameter_HMC(epsilon, parameterVal_paramNr, noisy_data, model, coords, iterations, sigma, filename, downsample_step, steps)
    return parameter_value_arr, parameter_name, Chi2_array, emission_array

#Function to get noisy data
def get_noisy_data(parameterVal_paramNr_copy, model, dates, downsample_step, max, sigma_noise):
    if len(parameterVal_paramNr_copy[1][0]) < 1:
        parameterVal_paramNr_copy = default_param(parameterVal_paramNr_copy, model)
    print(f"\n")
    print(f"Making Noisy data")
    comm_tod = TODLoader("/mn/stornext/d23/cmbco/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix')
    nside = 512 #hp.npix2nside(pix)
    lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
    #lon, lat = hp.pix2ang(nside, pix, lonlat=True) 
    #print(f"len(pix) = {len(pix)}")
    #print(f"len(pix[0:len(pix)//10]) = {len(pix[0:len(pix)//10])}")
    if len(dates) > 1:
        coords_arr = []
        for i in range(len(dates)):
            t0 = Time(dates[i]) #MJD
            dt = TimeDelta(1, format="sec")
            #Have t0  at different days
            obstimes = t0 + jnp.arange(lat.size) * dt
            coords = SkyCoord(
                    lon,
                    lat, 
                    unit=u.deg, 
                    frame="galactic", #Need to put in Galactic coordinates?
                    obstime=obstimes #Need array of obstimes 
            )
            coords_arr = np.vstack([coords_arr, coords])
    else:
        t0 = Time(dates[0]) #MJD
        dt = TimeDelta(1, format="sec")
        #Have t0  at different days
        obstimes = t0 + jnp.arange(lat.size) * dt
        coords = SkyCoord(
                lon,
                lat, 
                unit=u.deg, 
                frame="galactic", #Need to put in Galactic coordinates?
                obstime=obstimes #Need array of obstimes 
        )
        coords_arr = [coords]
    if any(isinstance(x, str) for sublist in parameterVal_paramNr_copy[1] for x in np.ravel(sublist)):
        #print(f"parameterVal_paramNr = {parameterVal_paramNr}")
        param_strings = parameterVal_paramNr_copy[1]
        #print(f"param_strings = {param_strings}")
        parameterVal_paramNr_copy[1] = parameter_list_str_to_nmbr(parameterVal_paramNr_copy[1], model)
        #print(f"Updated parameterVal_paramNr = {parameterVal_paramNr_copy}")
    #If they are sent in as numbers make an array that has a list of the strings
    if any(isinstance(x, (int, float, np.integer, np.floating)) for sublist in parameterVal_paramNr_copy[1] for x in np.ravel(sublist)):
        param_strings = parameter_nmbr_to_str_list(parameterVal_paramNr_copy[1] , model)

    #print(f"param_strings = {param_strings}")
    #print(f"parameterVal_paramNr_copy = {parameterVal_paramNr_copy}")

    if np.shape(parameterVal_paramNr_copy[0]) != np.shape(parameterVal_paramNr_copy[1][2]) or len(flatten_list(parameterVal_paramNr_copy[0])) < 1:
        print(f"Parameter array that was sent in is a different shape from parameter number array")
        print(f"Using default values for the given parameter number array")
        """
        parameters = model.get_parameters()
        default_parameterVal = np.zeros_like(parameterVal_paramNr_copy[1][2], dtype='float64')
        for i in range(len(parameterVal_paramNr_copy[1][0])):
            for j in range(len(parameterVal_paramNr_copy[1][1][i])):
                for k in range(len(parameterVal_paramNr_copy[1][2][i][j])):
                    default_parameterVal[i][j][k] = parameters[param_strings[0][i]][param_strings[1][i][j]][param_strings[2][i][j][k]] 
        #Change into default values
        #print(f"default_parameterVal = {default_parameterVal}")            
        parameterVal_paramNr_copy[0] = default_parameterVal
        """
        parameterVal_paramNr_copy[0] = copy.deepcopy(get_default_parameter_values(parameterVal_paramNr_copy, param_strings, model))
    #print(f"Test 1")
    parameter_nmbr_list = parameter_list_str_to_nmbr(param_strings, model)
    for i in range(len(coords_arr)):
        #print("Test 2")
        #print(f"parameterVal_paramNr_copy[0] = {parameterVal_paramNr_copy[0]}")
        #print(f"parameter_nmbr_list = {parameter_nmbr_list}")
        #print(f"parameterVal_paramNr_copy[0] = {parameterVal_paramNr_copy[0]}")
        #print(f"parameter_nmbr_list = {parameter_nmbr_list}")
        emission_og = get_emission(parameterVal_paramNr_copy[0], parameter_nmbr_list, max, coords_arr[i], downsample_step) #BE CAREFUL YOU NEED TO MAKE SURE YOU HAVE tHE RIGHT EMISSION
        noisy_data = noise_maker(emission_og, 0, sigma_noise) #(emission, mu, sigma)
    print(f"Noisy data made")
    print(f"\n")
    return noisy_data, emission_og
#Function to compare and find gaussian noise
def gauss_check(OG_emission, mean_parameter_paramName, model, dates, downsample_step, max):
    if len(mean_parameter_paramName[1][0]) < 1:
        mean_parameter_paramName = default_param(mean_parameter_paramName, model)
    comm_tod = TODLoader("/mn/stornext/d23/cmbco/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix') #size 688283
    #print(f"len(pix) = {len(pix)}")
    #print(f"len(pix[0:len(pix)//10]) = {len(pix[0:len(pix)//10])}")
    nside = 512 #hp.npix2nside(pix)
    lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
    #lon, lat = hp.pix2ang(nside, pix, lonlat=True)
    if len(dates) > 1:
        coords_arr = []
        for i in range(len(dates)):
            t0 = Time(dates[i]) #MJD
            dt = TimeDelta(1, format="sec")
            #Have t0  at different days
            obstimes = t0 + jnp.arange(lat.size) * dt
            coords = SkyCoord(
                    lon,
                    lat, 
                    unit=u.deg, 
                    frame="galactic", #Need to put in Galactic coordinates?
                    obstime=obstimes #Need array of obstimes 
            )
            coords_arr = np.vstack([coords_arr, coords])
    else:
        t0 = Time(dates[0]) #MJD
        dt = TimeDelta(1, format="sec")
        #Have t0  at different days
        obstimes = t0 + jnp.arange(lat.size) * dt
        coords = SkyCoord(
                lon,
                lat, 
                unit=u.deg, 
                frame="galactic", #Need to put in Galactic coordinates?
                obstime=obstimes #Need array of obstimes 
        )
        coords_arr = [coords]
    if any(isinstance(x, str) for sublist in mean_parameter_paramName[1] for x in np.ravel(sublist)):
        param_strings = mean_parameter_paramName[1]
        mean_parameter_paramName[1] = parameter_list_str_to_nmbr(mean_parameter_paramName[1], model)
    #If they are sent in as numbers make an array that has a list of the strings
    if any(isinstance(x, (int, float, np.integer, np.floating)) for sublist in mean_parameter_paramName[1] for x in np.ravel(sublist)):
        param_strings = parameter_nmbr_to_str_list(mean_parameter_paramName[1] , model)

    parameter_nmbr_list = parameter_list_str_to_nmbr(param_strings, model)
    for i in range(len(coords_arr)):
        data = get_emission(mean_parameter_paramName[0], parameter_nmbr_list, max, coords_arr[i], downsample_step) 

    r = OG_emission - data
    return r
#Function that plots the intensity against the time
def Intensity_plot(Intensity, filename):
    plt.plot(Intensity, label="Intensity")
    plt.ylabel(f"Emission[Mjy/sr]")
    plt.xlabel(f"Observations")
    plt.legend()
    plt.savefig(f"{filename}_emission.png")
    plt.close()

#Function that plots the healpix and compares the values in healpix
def healpix(Data_parameterVal_paramNr, calculated_parameterVal_paramNr, date):
    print(f"Making the healpix")

    #Convert 
    #Checks if there are any strings that are sent in
    if any(isinstance(x, str) for sublist in Data_parameterVal_paramNr[1] for x in np.ravel(sublist)):
        Data_parameterVal_paramNr[1] = parameter_list_str_to_nmbr(Data_parameterVal_paramNr[1], model)
        print(f"Changed the strings into integers for JAX readability for data")
    #Checks if there are any strings that are sent in
    if any(isinstance(x, str) for sublist in calculated_parameterVal_paramNr[1] for x in np.ravel(sublist)):
        calculated_parameterVal_paramNr[1] = parameter_list_str_to_nmbr(calculated_parameterVal_paramNr[1], model)
        print(f"Changed the strings into integers for JAX readability for calculated data")
    #Making the coords for the model
    nside = 256
    pixels = np.arange(hp.nside2npix(nside))
    lon, lat = hp.pix2ang(nside, pixels, lonlat=True)
    t0 = Time(f"{date}")
    dt = TimeDelta(1, format="sec")
    obstimes = t0 + np.arange(lat.size) * dt
    coords = SkyCoord(
            lon,
            lat, 
            unit=u.deg, 
            frame="galactic", #Need to put in Galactic coordinates?
            obstime=obstimes #Need array of obstimes 
        )
    
    #No need to downsample since we just want to plot the image
    downsample_step = 1 
    max = len(pixels) #len(pixels) = 786432
    data_emission = get_emission(Data_parameterVal_paramNr[0], Data_parameterVal_paramNr[1], max, coords, downsample_step)

    calculated_emission = get_emission(calculated_parameterVal_paramNr[0], calculated_parameterVal_paramNr[1], max, coords, downsample_step)

    difference = data_emission - calculated_emission
    print(f"max difference = {np.max(difference)}")
    print(f"min difference = {np.min(difference)}")
    # Create figure
    fig = plt.figure(figsize=(8, 18))
    fig.suptitle(f"Zodiacal light at 30 µm ({dates})", fontsize=16, fontweight="bold")

    # HEALPix Mollview projections
    hp.mollview(data_emission, unit="MJy/sr", cmap="afmhot",
                min=0, max=80, title="Data model", sub=(4, 1, 1))

    hp.mollview(calculated_emission, unit="MJy/sr", cmap="afmhot",
                min=0, max=80, title="Calculated data model", sub=(4, 1, 2))

    hp.mollview(difference, unit="MJy/sr", cmap="afmhot",
                min=np.min(difference).value, max=np.max(difference).value, title="Difference", sub=(4, 1, 3))
    
    #Rounding
    min = np.min(difference/data_emission*100).value
    max = np.max(difference/data_emission*110).value
    significant_fig = 1
    relative_min = round(min, significant_fig - int(np.floor(np.log10(abs(min)))) - 1)
    relative_max = round(max, significant_fig - int(np.floor(np.log10(abs(max)))) - 1)
    print(f"relative_min = {relative_min}")
    print(f"relative_max = {relative_max}")
    hp.mollview(difference/data_emission*100, cmap="afmhot",
                min=relative_min, max=relative_max, title="Relative Difference", sub=(4, 1, 4))

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(f"Images_6/healpix_map_{len(flatten_list(calculated_parameterVal_paramNr[0]))}_parameters.png")
    plt.close()
    print(f"Finished making the healpix plot")
"""
#Function that plots the number density
def number_density_plot(model, Plane, Date, animation=False):
    N = 200
    x = np.linspace(-5, 5, N) * u.AU  # x-plane
    y = np.linspace(-5, 5, N) * u.AU  # y-plane
    z = np.linspace(-2, 2, N) * u.AU  # z-plane
    Figure = plt.figure()
    if animation==True:
        def AnimationFunction(frame): 
            multi_param_change(model, parameter_str_list, new_parameter_values) 

            density_grid = zodipy.grid_number_density(x, y, z, #density_grid has shape (6, 200, 200, 200)                                                                                   
            obstime=Time(f"{Date}", scale="utc"), model=model._ipd_model,)
            density_grid = density_grid.sum(axis=0)  # Sum over all components
            # [N // 2,:,:] yz-plane
            # [:,N // 2,:] xz-plane
            # [:,:,N // 2] xy-plane
            if Plane == "yz":
                plt.pcolormesh(x, y, density_grid[N // 2,:,:].T,  # cross section in the yz-plane
                cmap="afmhot",
                norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
                shading="gouraud",
                rasterized=True,)
                plt.xlabel("x [AU]")
                plt.ylabel("z [AU]")

            if Plane == "xz":
                plt.pcolormesh(x, y, density_grid[:,N // 2,:].T,  # cross section in the xz-plane
                cmap="afmhot",
                norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
                shading="gouraud",
                rasterized=True,)
                plt.xlabel("x [AU]")
                plt.ylabel("y [AU]")

            if Plane == "xy": 
                plt.pcolormesh(x, y, density_grid[:,:,N // 2].T,  # cross section in the xy-plane
                cmap="afmhot",
                norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
                shading="gouraud",
                rasterized=True,)
                plt.xlabel("y [AU]")
                plt.ylabel("z [AU]")

#"""
"""
#Function that checks if the sent in parameterVal_paramNr has any empty arrays and fills them in, kinda like a default
def default_param(parameterVal_paramNr, model):
    parameters = model.get_parameters()
    #print(f"parameters = {parameters}")
    top = list(parameters.keys())
    #Constructing the array to look have this inhomogenous shape:[ [], [[]], [[[]]] ]
    default_paramNr = [[], [[]], [[[]]]]
    if len(parameterVal_paramNr[1][0]) < 1:
        for i in range(len(top)):
            #print(f"default_paramNr[0] = {default_paramNr[0]}")
            #print(f"top[i] = {top[i]}")
            default_paramNr[0].append(top[i]) 
            #print(f"default_paramNr[0] = {default_paramNr[0]}")
            mid_params = []
            bottom_bottom_params = []
            if hasattr(parameters[top[i]], "keys"):
                mid = list(parameters[top[i]].keys())
            else:
                #print(f"parameters[top[i]] = {parameters[top[i]]}")
                if isinstance(parameters[top[i]], (int, float)): 
                    mid_params.append(parameters[top[i]])
                    #print(f"mid_params = {mid_params}")
                else:
                    mid_params.extend(list(parameters[top[i]]))
                    if mid_params:
                        default_paramNr[1].append(mid_params)
                continue
            for j in range(len(mid)):
                mid_params.append(mid[j])
                bottom_params = []
                #print(f"parameters[top[i]][mid[j]].keys() = {parameters[top[i]][mid[j]].keys()}")
                if hasattr(parameters[top[i]][mid[j]], "keys"):
                    bottom = list(parameters[top[i]][mid[j]].keys())
                else:
                    bottom_params.extend(list(parameters[top[i]][mid[j]]))
                    if bottom_params:
                        default_paramNr[2].append(bottom_params)   
                    continue
                for k in range(len(bottom)):
                    bottom_params.append(bottom[k])
                if bottom_bottom_params:
                    bottom_bottom_params.append(bottom_params)
            default_paramNr[1] = [x for x in default_paramNr[1] if x]  
            default_paramNr[2] = [x for x in default_paramNr[2] if x] 
        print(f"default_paramNr = {default_paramNr}")
        print(f"")
        return default_paramNr
"""
def default_param(parameterVal_paramNr, model):
    parameters = model.get_parameters()
    #print(parameters)
    print(f"{parameters['spectrum']}")
    parameters['spectrum'] = [  1,  1,   1,    1,   1,    1,    1,   1,   1,   1  ]*u.micron
    model.update_parameters(parameters)
    print(f"{parameters['spectrum']}")
    print(f"")
    top = list(parameters.keys())

    # Initialize the result structure
    default_paramNr = [[], [[]], [[[]]]]

    # Fill in the top-level keys
    if len(parameterVal_paramNr[1][0]) < 1:
        for i in range(len(top)):
            # Append top-level keys
            default_paramNr[0].append(top[i]) 

            # Mid-level parameters (if the key has nested dictionaries)
            mid_params = []
            if hasattr(parameters[top[i]], "keys"):  # It's a dictionary
                mid = list(parameters[top[i]].keys())
                for j in range(len(mid)):
                    mid_params.append(mid[j])

                    bottom_params = []
                    if hasattr(parameters[top[i]][mid[j]], "keys"):  # Check for further nested dictionaries
                        bottom = list(parameters[top[i]][mid[j]].keys())
                        for k in range(len(bottom)):
                            bottom_params.append(bottom[k])
                        default_paramNr[2].append(bottom_params)  # Store bottom-level keys
                    else:
                        # If it's a value (not a dictionary), append it to the mid-level
                        mid_params.append(parameters[top[i]][mid[j]])

                default_paramNr[1].append(mid_params)  # Store mid-level values
            else:
                # If it's not a dictionary, append directly to the mid-level
                default_paramNr[1].append([parameters[top[i]]])

        # Clean empty sublists
        default_paramNr[1] = [x for x in default_paramNr[1] if x]
        default_paramNr[2] = [x for x in default_paramNr[2] if x] 

    # Print the final structure for debugging
    #print(f"default_paramNr = {default_paramNr}")
    
    return default_paramNr






test_default = [
    # [0] Top Keys
    ["comps", "spectrum", "T_0", "delta", "emissivities", "albedos", "solar_irradiance", "C1", "C2", "C3"],

    # [1] Middle Keys (Grouped per Top Key)
    [
        # Middle keys of "comps"
        ["cloud", "band1", "band2", "band3", "ring", "feature"],
        # "spectrum" has no middle keys; its value is included
        ["<Quantity [1.25, 2.2, 3.5, 4.9, 12.0, 25.0, 60.0, 100.0, 140.0, 240.0] micron>"],
        # "T_0" has no middle keys; its value is included
        [286],
        # "delta" has no middle keys; its value is included
        [0.46686259861486573],
        # Middle keys of "emissivities"
        ["cloud", "band1", "band2", "band3", "ring", "feature"],
        # Middle keys of "albedos"
        ["cloud", "band1", "band2", "band3", "ring", "feature"],
        # "solar_irradiance" has no middle keys; its value is included
        [234056060.0, 123098740.0, 64292872.0, 35733824.0, 5763843.0, 1327989.4, 230553.73, 82999.336, 42346.605, 14409.608],
        # "C1" has no middle keys; its value is included
        [-0.94209999, -0.52670002, -0.4312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "C2" has no middle keys; its value is included
        [0.1214, 0.18719999, 0.1715, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "C3" has no middle keys; its value is included
        [-0.1648, -0.59829998, -0.63330001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],

    # [2] Bottom Keys (Grouped per Middle Key)
    [
        # Bottom keys of each middle key in "comps"
        [
            # Bottom keys of "cloud"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "alpha", "beta", "gamma", "mu"],
            # Bottom keys of "band1"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "delta_zeta", "v", "p", "delta_r"],
            # Bottom keys of "band2"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "delta_zeta", "v", "p", "delta_r"],
            # Bottom keys of "band3"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "delta_zeta", "v", "p", "delta_r"],
            # Bottom keys of "ring"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "R", "sigma_r", "sigma_z"],
            # Bottom keys of "feature"
            ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "R", "sigma_r", "sigma_z", "theta", "sigma_theta"]
        ],
        # "spectrum" has no bottom keys; its value is included directly
        ["<Quantity [1.25, 2.2, 3.5, 4.9, 12.0, 25.0, 60.0, 100.0, 140.0, 240.0] micron>"],
        # "T_0" has no bottom keys; its value is included directly
        [286],
        # "delta" has no bottom keys; its value is included directly
        [0.46686259861486573],
        # Bottom keys of each middle key in "emissivities"
        [
            # Bottom keys of "cloud"
            [1.0, 1.0, 1.659892404064974, 0.9974090848665298, 0.9576691480594887, 1.0, 0.7333883261676887, 0.6478988180222407, 0.6769420588104739, 0.5191208540195074],
            # Bottom keys of "band1"
            [1.0, 1.0, 1.659892404064974, 0.3592645195835044, 1.0127926948497732, 1.0, 1.2539242027824944, 1.5167023376593836, 1.1317240279481993, 1.3996145963796358],
            # Bottom keys of "band2"
            [1.0, 1.0, 1.659892404064974, 0.3592645195835044, 1.0127926948497732, 1.0, 1.2539242027824944, 1.5167023376593836, 1.1317240279481993, 1.3996145963796358],
            # Bottom keys of "band3"
            [1.0, 1.0, 1.659892404064974, 0.3592645195835044, 1.0127926948497732, 1.0, 1.2539242027824944, 1.5167023376593836, 1.1317240279481993, 1.3996145963796358],
            # Bottom keys of "ring"
            [1.0, 1.0, 1.659892404064974, 1.0675116768340536, 1.0608768682182081, 1.0, 0.8726636137878518, 1.0985346556794289, 1.1515825707787077, 0.8576380099421744],
            # Bottom keys of "feature"
            [1.0, 1.0, 1.659892404064974, 1.0675116768340536, 1.0608768682182081, 1.0, 0.8726636137878518, 1.0985346556794289, 1.1515825707787077, 0.8576380099421744]
        ],
        # Bottom keys of each middle key in "albedos"
        [
            # Bottom keys of "cloud"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Bottom keys of "band1"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Bottom keys of "band2"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Bottom keys of "band3"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Bottom keys of "ring"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # Bottom keys of "feature"
            [0.20411939612669797, 0.255211328920523, 0.21043660481632315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
        # "solar_irradiance" has no bottom keys; its value is included directly
        [234056060.0, 123098740.0, 64292872.0, 35733824.0, 5763843.0, 1327989.4, 230553.73, 82999.336, 42346.605, 14409.608],
        # "C1" has no bottom keys; its value is included directly
        [-0.94209999, -0.52670002, -0.4312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "C2" has no bottom keys; its value is included directly
        [0.1214, 0.18719999, 0.1715, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # "C3" has no bottom keys; its value is included directly
        [-0.1648, -0.59829998, -0.63330001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
]

"""
print(f"len(test_default) = {len(test_default)}")
print(f"len(test_default[0]) = {len(test_default[0])}")
print(f"len(test_default[1]) = {len(test_default[1])}")
print(f"len(test_default[2]) = {len(test_default[2])}")
#"""
parameters = [[[]]] #testing default function
parameter_str = [[], [[]], [[[]]]] #Send in empty to get all the parameters
model = zodipy.Model(30 * u.micron)
test_parameterVal_paramNr = [parameters, parameter_str]
#default = default_param(test_parameterVal_paramNr, model)
"""
#print(f"default = {default}")
print(f"len(default) = {len(default)}")
print(f"len(default[0]) = {len(default[0])}")
#print(f"default[0] = {default[0]}")
print(f"len(default[1]) = {len(default[1])}")
#print(f"default[1] = {default[1]}")
print(f"len(default[2]) = {len(default[2])}")
#print(f"default[2] = {default[2]}")
#print(f"default = {default}")
#"""
#Testing out the code
#epsilon = 3.2e-06
#epsilon = 1.2211162399298152e-05
#epsilon = 0.03776484165282503
#epsilon = 0.043441701276370086 #omega 10 steps 0.7 accept ratr
epsilon = 4.978439260196648e-12 #For Omega and N_0
#1.2211162399298152e-05 give 0.25
#Find the minimized value
#parameters = [[[77.65795555409711, 1.134437388142796e-07]]] #testing default function (Success)
#True_parameters = [[[77.65795555409711, 1.134437388142796e-07]]]
#True_parameters = [[[77.65795555409711]]]
#True_parameters = [[[77.65795555409711, 1.337069670593028]]]
True_parameters = [[[0.011887800744346281, 0.005476506466226378, -0.0021530908020710744, 2.033518807239077, 77.65795555409711, 1.134437388142796e-07, 1.337069670593028, 4.141500415758664, 0.9420617939335804, 0.1887317648909019]]]
#True_parameters = [[[77.65795555409711, 1.337069670593028, 0.9420617939335804]]]
parameters = [[[]]]
#parameter_str = [["comps"], [["cloud"]], [[["x_0", "y_0", "z_0", "i", "Omega", "alpha", "beta", "gamma", "mu"]]]] #Send in empty to get all the parameters no n_0
parameter_str = [["comps"], [["cloud"]], [[["x_0", "y_0", "z_0", "i", "Omega", "n_0","alpha", "beta", "gamma", "mu"]]]]
#parameter_str = [["comps"], [["cloud"]], [[[]]]] #To test default (Success)
#parameter_str = [["comps"], [["cloud"]], [[["Omega"]]]]
#parameter_str = [["comps"], [["cloud"]], [[["Omega", "alpha", "gamma"]]]]
#parameter_str = [["comps"], [["cloud"]], [[["Omega", "alpha"]]]]
#parameter_str = [["comps"], [["cloud"]], [[["Omega", "n_0"]]]] #To test 2 parameter (Success)
#parameter_str = [["delta"], [[]], [[[]]]] #To test if there is a float (Success)
#parameter_str = [["T_0"], [[]], [[[]]]] #To test if there is a int (Success)
#parameter_str = [["C3"], [[]], [[[]]]] #To test if there is an array/list instead of a singular value (Success??? All of the values were accepted during the tuning phase)
#parameter_str = [["emissivities"], [["feature"]], [[[]]]] #To test if there is a tuple of values
#parameter_str = [[], [[]], [[[]]]] #The ultimate test POSTPONE
parameterVal_paramNr = [np.asarray(parameters), parameter_str]
True_parameterVal_paramNr = [np.asarray(True_parameters), parameter_str]
#truth_dict = {"Omega":77.65795555409711, "alpha":1.337069670593028, "Chi2":6940}
#truth_dict = {"Omega":77.65795555409711, "Chi2":6940}
#truth_dict = {"Omega":77.65795555409711, "n_0": 1.134437388142796e-07, "Chi2":6940}
#truth_dict = {"Omega":77.65795555409711, "alpha":1.337069670593028, "gamma":0.9420617939335804, "Chi2":6940}
#"""
truth_dict = {'x_0': 0.011887800744346281, 
'y_0': 0.005476506466226378, 
'z_0': -0.0021530908020710744, 
'i': 2.033518807239077, 
'Omega': 77.65795555409711, 
'n_0': 1.134437388142796e-07, 
'alpha': 1.337069670593028, 
'beta': 4.141500415758664, 
'gamma': 0.9420617939335804, 
'mu': 0.1887317648909019, "Chi2":6900}
#"""
#truth_dict = {"Omega":77.65795555409711, "Chi2":6940}
truth = list(truth_dict.values())
model = zodipy.Model(30 * u.micron)
downsample_step = 10

dates = ["2022-01-14"]

iterations = 10000 #Important, need enough iterations so the values have some variance for corner.

steps = 10 #IMPORTANT TO HAVE 10 STEPS AS THAT'S WHAT I USED DURING TUNING

sigma_noise = 0.1 #sigma of the noisy data set we compare with

#sigma = [[[1.e-07, 1.e-07, 1.e-06, 1.e-06, 1.e-05, 1.e-06, 1.e-06, 1.e-06, 1.e-07]]] #Without n_0
filename = f"HMC_cloud_{len(truth_dict) - 1}_parameters_{iterations}_iterations_{sigma_noise}_noise_{steps}_integration_step_size"

max = 68828
#max = 688283
True_parameterVal_paramNr_copy = copy.deepcopy(True_parameterVal_paramNr)
#"""

noisy_data, OG_emission = get_noisy_data(True_parameterVal_paramNr_copy, model, dates, downsample_step, max, sigma_noise)
#"""
#Put sigma_file to False if there are no files
sigma_file = False#f"Images_3/{filename}_sigma_10parameters.txt"

#parameter_value_arr, parameter_name, Chi2_array, emission_array = get_parameter_HMC_function(epsilon, parameterVal_paramNr, noisy_data, model, dates, iterations, filename, downsample_step, sigma_file, steps)

mean_val, mean_parameters = get_corner_plt_readfile(f"Images_6/{filename}.txt", filename, truth)
print(f"mean_val = {mean_val}")
#print(f"mean_paramters = {mean_paramters}")
mean_parameter_paramName = [np.asarray([[mean_parameters]]), parameter_str]
r = gauss_check(OG_emission, mean_parameter_paramName, model, dates, downsample_step, max)

#print(f"OG_emission.shape = {OG_emission.shape}")

#mean_parameters = [0., 0.]
Data_parameterVal_paramNr = copy.deepcopy(True_parameterVal_paramNr)
calculated_parameterVal_paramNr = [np.asarray([[mean_parameters]]), parameter_str]
healpix(Data_parameterVal_paramNr, calculated_parameterVal_paramNr, dates[0])
#"""


"""
parameters = model.get_parameters()
top = list(parameters.keys())
print(f"Top {top}")
middle = list(parameters[top[0]].keys())
#print(f"middle = {middle}")
bottom = list(parameters[top[0]][middle[0]].keys())
#print(f"Bottom = {bottom}")
value = parameters[top[0]][middle[0]][bottom[0]]
#print(value)

print(parameters["emissivities"]["feature"])
# (1.0, 1.0, 1.659892404064974, 1.0675116768340536, 1.0608768682182081, 1.0, 0.8726636137878518, 1.0985346556794289, 1.1515825707787077, 0.8576380099421744)
parameters["emissivities"]["feature"] = [1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 0.1, 1.1, 1.1, 0.1]
print(parameters['comps'].keys())
print(parameters['spectrum']) #array
print(f"T_0 = {parameters['T_0']}") #int value
print(f"delta = {parameters['delta']}") #float value
print(f"emissivites.keys(): {parameters['emissivities'].keys()}")
print(f"albedos.keys():{parameters['albedos'].keys()}")
print(f"albedos = {parameters['albedos']}")
print(f"solar_irradiance = {parameters['solar_irradiance']}") #array
print(f"C1 = {parameters['C1']}") #array
print(f"C2 = {parameters['C2']}") #array
print(f"C3 = {parameters['C3']}") #array
#"""


"""
#Debugging test
alpha_true = 1.337069670593028
omega_true = 77.65795555409711
n_0_true = 1.134437388142796e-07 
omega = np.linspace(77.6, 80, 100)
alpha = np.linspace(1.337, 1.35, 50)
n_0 = np.linspace(1.13443738e-07 , 1.13443739e-07 , 100)
chi2_array = np.zeros(len(n_0))
for i in tqdm(range(len(n_0))):
    parameters = [[[n_0[i]]]]
    parameter_str = [["comps"], [["cloud"]], [[["n_0"]]]]
    #parameters = [[[77.65795555409711, n_0[i]]]]
    #parameter_str = [["comps"], [["cloud"]], [[["Omega", "n_0"]]]]
    parameter_int = parameter_list_str_to_nmbr(parameter_str, model)
    comm_tod = TODLoader("/mn/stornext/d23/cmbco/cg/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix')
    nside = 512 #hp.npix2nside(pix)
    lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
    t0 = Time(dates[0]) #MJD
    dt = TimeDelta(1, format="sec")
    #Have t0  at different days
    obstimes = t0 + jnp.arange(lat.size) * dt
    coords = SkyCoord(
                lon,
                lat, 
                unit=u.deg, 
                frame="galactic", #Need to put in Galactic coordinates?
                obstime=obstimes #Need array of obstimes 
        )
    chi2_array[i] = get_chi2(parameters, parameter_int, max, coords, noisy_data, downsample_step)
    #print(f"n_0 = {n_0[i]} with chi2 = {chi2_array[i]} ")
    if chi2_array[i] > 10000 and chi2_array[i] < 12000:
        print(f"n_0 = {n_0[i]} with chi2 = {chi2_array[i]} ")
plt.plot(n_0, chi2_array)
plt.axvline(n_0_true)
plt.title(f"Chi2 values")
plt.ylabel(f"Chi2")
plt.xlabel(f"n_0")
plt.savefig(f"Chi2_test_n_0_{sigma_noise}_noise.png")
plt.close()
#"""
"""
filename = f"Images_3/HMC_cloud_Omega_alpha_test_700_iterations_0.001_noise.txt"
with open(filename, 'r') as file:
    labels = file.readline().strip().split()
data = np.loadtxt(filename, skiprows = 1)
iterations = len(data[:,-1])
omega = data[:,0]
alpha = data[:,1]
plt.plot(omega[300:])
plt.title("omega")
plt.ylabel(f"Omega")
plt.xlabel(f"iterations")
plt.savefig(f"omega_iterations.png")
plt.close()

plt.plot(alpha[300:])
plt.title("alpha")
plt.ylabel(f"alpha")
plt.xlabel(f"iterations")
plt.savefig(f"alpha_iterations.png")
plt.close()
#"""

"""
omega_true = 77.65795555409711 #Chi2 = 10000 [76, 79]
alpha_true = 1.337069670593028 #Chi2 = 10000 [1.332, 1.342]
n_0_true = 1.134437388142796e-07 #Chi2 = 10000 [1.1324e-07, 1.1364e-07]
size = 10
omega = np.linspace(76, 79, size)
alpha = np.linspace(1.332, 1.342, size)
#n_0 = np.linspace(1.1324e-07, 1.1364e-07 , size)
n_0 = np.linspace(1.13443e-07, 1.13444e-07 , size)
chi2_array = np.zeros((len(omega), len(n_0)))
for i in tqdm(range(len(omega))):
    for j in range(len(n_0)):
        parameters = [[[omega[i], n_0[j]]]]
        parameter_str = [["comps"], [["cloud"]], [[["Omega", "n_0"]]]]
        parameter_int = parameter_list_str_to_nmbr(parameter_str, model)
        comm_tod = TODLoader("/mn/stornext/d23/cmbco/cg/dirbe/data", "DIRBE")
        comm_tod.init_file('05_nside512_V23', '')
        pix = comm_tod.load_field(f'000185/05_A/pix')
        nside = 512 #hp.npix2nside(pix)
        lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
        t0 = Time(dates[0]) #MJD
        dt = TimeDelta(1, format="sec")
        #Have t0  at different days
        obstimes = t0 + jnp.arange(lat.size) * dt
        coords = SkyCoord(
                    lon,
                    lat, 
                    unit=u.deg, 
                    frame="galactic", #Need to put in Galactic coordinates?
                    obstime=obstimes #Need array of obstimes 
            )
        chi2_array[i][j] = get_chi2(parameters, parameter_int, max, coords, noisy_data, downsample_step)
        #print(f"chi2_array[i][j] = {chi2_array[i][j] }")
np.savetxt(f"Omega_n_0_Chi2_{size}x{size}.txt", chi2_array)
plt.imshow(chi2_array,  extent=[omega.min(), omega.max(), n_0.min(), n_0.max()], origin='lower', cmap='viridis', aspect='auto', vmax = 10000)
plt.colorbar(label="Chi2")
plt.axvline(omega_true, color = "red", label="True Omega")
plt.axhline(n_0_true, color = "orange", label=" True N_0")
plt.xlabel("Omega")
plt.ylabel("n_0")
plt.title(f"Omega/n_0 {sigma_noise} noise")
plt.legend()
plt.savefig(f"Chi2_imshow_n_0_Omega_{sigma_noise}_noise.png")
plt.close()
plt.plot()
#"""
"""
def mean_std(omega, alpha, chi2):
    #print(f"Chi2 = {chi2}")
    P = np.exp(-(chi2 - np.max(chi2))/2)
    #print(f"chi2 - np.max(chi2) = {chi2 - np.max(chi2)}")
    #print(f"P = {P}")
    P = P/np.sum(P)
    omega_mean = np.sum(omega*P)
    omega_mean2 = np.sum(omega**2*P)
    omega_std = np.sqrt(omega_mean2 - omega_mean**2)

    alpha_mean = np.sum(alpha*P)
    alpha_mean2 = np.sum(alpha**2*P)
    alpha_std = np.sqrt(alpha_mean2 - alpha_mean**2)

    return omega_mean, omega_std, alpha_mean, alpha_std

#filename = f"Images_2/HMC_cloud_Omega_alpha_test_700_iterations_0.001_noise.txt"
with open(filename, 'r') as file:
    labels = file.readline().strip().split()
data = np.loadtxt(filename, skiprows = 1)
iterations = len(data[:,-1])
omega_data = data[:,0]
n_0_data = data[:,1]
chi2_data = data[:,-1]
chi2_array_100x100 = np.loadtxt("Omega_n_0_Chi2_100x100.txt")

omega_true = 77.65795555409711 #Chi2 = 10000 [76, 79]
alpha_true = 1.337069670593028 #Chi2 = 10000 [1.332, 1.342]
n_0_true = 1.134437388142796e-07 #Chi2 = 100000 [???] 
#plt.plot(alpha_data)
sigma_noise = 0.001
fig, axes = plt.subplots(sharex=True, nrows=2)
axes[0].plot(n_0_data)
axes[1].plot(omega_data)
axes[0].set_ylabel("n_0")
axes[1].set_ylabel("Omega")
plt.title(f"{iterations} iterations with {sigma_noise} noise")
plt.savefig("Omega_n_0_subplots.png")
plt.close()
#"""
"""
plt.plot(omega_data[:500], alpha_data[:500])
plt.imshow(chi2_array_100x100,  extent=[omega_data.min(), omega_data.max(), alpha_data.min(), alpha_data.max()], origin='lower', cmap='viridis', aspect='auto', vmax = 10000)
plt.xlabel(f"Omega")
plt.ylabel(f"alpha")
plt.colorbar(label="Chi2")
plt.axvline(omega_true, color = "red", label="True Omega")
plt.axhline(alpha_true, color = "orange", label=" True Alpha")
plt.title(f"Parameter path")
plt.savefig(f"Omega_alpha_100_iterations.png")
plt.close()
"""
"""
omega_mean_data, omega_std_data, alpha_mean_data, alpha_std_data = mean_std(omega_data, alpha_data, chi2_data)
print(f"omega_mean data= {omega_mean_data}")
print(f"omega_std data= {omega_std_data}")
print(f"alpha_mean data= {alpha_mean_data}")
print(f"alpha_std data= {alpha_std_data}")

omega_mean, omega_std, alpha_mean, alpha_std = mean_std(omega, alpha, chi2_array)
print(f"omega_mean baseline= {omega_mean}")
print(f"omega_std baseline= {omega_std}")
print(f"alpha_mean baseline= {alpha_mean}")
print(f"alpha_std baseline= {alpha_std}")

#"""
#Function that gives emission from TOD :38s
def get_emission_2(model):
    comm_tod = TODLoader("/mn/stornext/d23/cmbco/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix')
    nside = 512 #hp.npix2nside(pix)
    lon, lat = hp.pix2ang(nside, pix[0:len(pix)//10], lonlat=True) 
    t0 = Time("2022-01-14")
    dt = TimeDelta(1, format="sec")
    obstimes = t0 + jnp.arange(lat.size) * dt
    coords = SkyCoord(
            lon,
            lat, 
            unit=u.deg, 
            frame="galactic", #Need to put in Galactic coordinates?
            obstime=obstimes #Need array of obstimes 
        )
    #time0 = time.time()
    emission = model.evaluate(coords, nprocesses=1) #~35.45s to run. 8.4s with 6 cores
    #time1 = time.time()
    #print(f"{time1-time0}s")
    Observations = jnp.linspace(0, len(obstimes), len(obstimes))
    return emission, Observations

def multi_param_change_2(model, new_param_values, category):
    parameters = model.get_parameters()
    for i in range(len(category)):
        parameters["comps"]["cloud"][category[i]] = new_param_values[i]
    model.update_parameters(parameters)

def get_chi2_2(data, model, sigma, with_jax=False):
    chi2 = 0
    emission, observations = get_emission_2(model)
    if with_jax==False:
        for i in range(len(data)):
            chi2 += ((data[i].value - emission[i].value)/sigma)**2
    else:
        chi2 = 10
        #for i in range(len(data)):   
    return chi2

#MHS function
def get_MHS_moreparams(noise_data, initial_values, step_size, size, category, name):
    model = zodipy.Model(30 * u.micron)
    param_arr = np.zeros((len(initial_values), size))
    param_prop = np.zeros(len(initial_values))
    chi2_arr = np.zeros(size)
    for i in range(len(initial_values)):
        param_arr[i][0] = initial_values[i]
    multi_param_change_2(model, initial_values, category)
    chi2_arr[0] = get_chi2_2(noise_data, model, 0.1, with_jax=False)
    rng = np.random.default_rng()
    accept_rate = 0
    if len(category) > 1:
        #Get Covariance matrix
        Covariance_matrix = find_coavriance(noise_data, initial_values, step_size, category)
        mean = np.zeros(len(category))
        n = np.random.multivariate_normal(mean, Covariance_matrix, size=size)
    else:
        n = np.random.normal(0, 0.1, (size, 1))
    #print(np.shape(n))
    #Metropolis Hastings
    for i in tqdm(range(1, size, 1)):
        for j in range(len(initial_values)):
            #n = np.random.normal(0, step_size[j]) #step size
            param_prop[j] = param_arr[j][i-1] + n[i][j]

        multi_param_change_2(model, param_prop, category)
        chi2_prop = get_chi2_2(noise_data, model, 0.1, with_jax=False)
        probability = (-(chi2_prop - chi2_arr[i-1])/2) 
        if probability > np.log(rng.random()):
            chi2_arr[i] = chi2_prop
            for j in range(len(initial_values)):
                param_arr[j][i] = param_prop[j]
            accept_rate += 1
        else:
            chi2_arr[i] = chi2_arr[i-1]
            for j in range(len(initial_values)):
                param_arr[j][i] = param_arr[j][i-1]
    
    print(f"{accept_rate} out of {size-1}")
    with open(f"Images_4/accept_rate_{size}_{name}_{len(initial_values)}_params.txt", 'w') as file:
        file.write(f"accept rate: {accept_rate} out of {size-1}\n")
    for i in range(len(initial_values)):
        plt.plot(param_arr[i][:])
        plt.title(f"Plot of {category[i]}")
        plt.savefig(f"Images_4/param_{category[i]}_iter_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.hist(param_arr[i][:])
        plt.title(f"Histogram for {category[i]}")
        plt.savefig(f"Images_4/param_{category[i]}_hist_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
        plt.close()
    plt.plot(chi2_arr)
    plt.title(r"Plot of $\chi^2$")
    plt.savefig(f"Images_4/Chi2_iter_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
    plt.close()
    #write values into a file
    for j in range(len(initial_values)):
        with open(f"Images_4/param_{category[j]}_chi_val_{size}_{name}.txt", 'w') as file:
            file.write(f"chi2\t{category[j]}\n")
            for i in range(len(chi2_arr)):
                file.write(f"{chi2_arr[i]}\t{param_arr[j][i]}\n")
    return chi2_arr, param_arr

#Covariance Function
def find_coavriance(noise_data, initial_values, step_size, category):
    size = 10000 #How many points normally 5000
    #Code from get_MHS_moreparams
    model = zodipy.Model(30 * u.micron)
    param_arr = np.zeros((len(initial_values), size))
    param_prop = np.zeros(len(initial_values))
    chi2_arr = np.zeros(size)
    for i in range(len(initial_values)):
        param_arr[i][0] = initial_values[i]
    multi_param_change_2(model, initial_values, category)
    chi2_arr[0] = get_chi2_2(noise_data, model, 0.1, with_jax=False)
    rng = np.random.default_rng()
    accept_rate = 0
    for i in tqdm(range(1, len(param_arr[0]), 1)):
        for j in range(len(initial_values)):
            n = np.random.normal(0, step_size[j]) #step size
            param_prop[j] = param_arr[j][i-1] + n
        multi_param_change_2(model, param_prop, category)
        chi2_prop = get_chi2_2(noise_data, model, 0.1, with_jax=False)
        probability = (-(chi2_prop - chi2_arr[i-1])/2) 
        if probability > np.log(rng.random()):
            chi2_arr[i] = chi2_prop
            for j in range(len(initial_values)):
                param_arr[j][i] = param_prop[j]
            accept_rate += 1
        else:
            chi2_arr[i] = chi2_arr[i-1]
            for j in range(len(initial_values)):
                param_arr[j][i] = param_arr[j][i-1]
    #Find covariance of the parameters
    covariance_matrix = np.cov(param_arr)
    print(f"Covariance matrix: {covariance_matrix}")
    with open(f"Images_4/Covariance_matrix_{size}.txt", 'w') as file:
        file.write(f"Covariance Matrix: {covariance_matrix}\n")
    return covariance_matrix
"""
'x_0': 0.011887800744346281, 
'y_0': 0.005476506466226378, 
'z_0': -0.0021530908020710744, 
'i': 2.033518807239077, 
'Omega': 77.65795555409711, 
'n_0': 1.134437388142796e-07, 
'alpha': 1.337069670593028, 
'beta': 4.141500415758664, 
'gamma': 0.9420617939335804, 
'mu': 0.1887317648909019
"""
#3 param
"""
init_val = [77.65795555409711, 1.337069670593028, 0.9420617939335804]
param =  ["Omega", "alpha", "gamma"]
stepsize = [0.1, 0.001, 0.001]
#"""
#2 param
"""
init_val = [77.65795555409711, 1.134437388142796e-07]
param =  ["Omega", "n_0"]
stepsize = [0.1, 0.001]
#"""
#1 param
#"""
init_val = [77.65795555409711]
param =  ["Omega"]
stepsize = [0.1]
#"""

size = 10000 #100k
#chi2_arr, param_arr = get_MHS_moreparams(noisy_data, init_val, stepsize, size, param, name="covariance_1_param")

#Timing the code
t1_timer = time.time()

total = t1_timer-t0_timer
print(f"The code took {total:.2f}s to run")
