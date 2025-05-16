"""
from cosmoglobe.tod_tools import TODLoader
comm_tod = TODLoader("/mn/stornext/d5/data/metins/dirbe/data", "DIRBE")
comm_tod.init_file('05_nside512_V23', '')
pix = comm_tod.load_field(f'000185/05_A/pix')
import healpy as hp
lon, lat = hp.pix2ang(512, pix, lonlat=True)
"""
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
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
np.random.seed(5)
t0 = time.time()

"""
mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
plt.savefig("test1")
plt.close()
mean = (1, 2)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (3, 3))
x.shape
cov = np.array([[6, 0], [0, 3.5]])
pts = np.random.multivariate_normal([0, 0], cov, size=800)
pts.mean(axis=0)
np.cov(pts.T)
np.corrcoef(pts.T)[0, 1]
plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
plt.xlim(10,-10)
plt.ylim(10,-10)
plt.grid()
plt.show()
plt.savefig("test3.png")
plt.close()

x = np.random.normal(0, np.sqrt(6), size=800) 
y = np.random.normal(0, np.sqrt(3.5), size=800) 
plt.plot(x, y, '.', alpha=0.5)
plt.grid()
plt.xlim(10,-10)
plt.ylim(10,-10)
plt.show()
plt.savefig("test4.png")
plt.close()

pts = np.random.multivariate_normal([0, 0], cov, size=4)
#print(pts)

covar = [[1.07253606e-02, 2.47843294e-05, 8.37030498e-05],[2.47843294e-05, 1.71749289e-07, 1.29550680e-07],
         [8.37030498e-05, 1.29550680e-07, 7.77591383e-07]]
mean = np.zeros(3)
n = np.random.multivariate_normal(mean, covar, size=1000)
"""
"""
figure, tax = ternary.figure(scale=0.1)
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=5)
tax.scatter(n, marker="o", color="blue", s=10)
tax.set_title("Triangle plot")
# Set Axis labels and Title
fontsize = 12
offset = 0.14
tax.right_corner_label("X", fontsize=fontsize)
tax.top_corner_label("Y", fontsize=fontsize)
tax.left_corner_label("Z", fontsize=fontsize)
tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize, offset=offset)
tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize, offset=offset)
tax.clear_matplotlib_ticks()
plt.savefig("triangle_plot_test.png")
plt.close()
"""
"""
fig, ax = plt.subplots(subplot_kw={'projection': 'ternary'})
ax.set_tlim(0, 1)  # Left axis
ax.set_llim(0, 1)  # Right axis
ax.set_rlim(0, 1)  # Bottom axis

# Scatter plot the data
left, right, bottom = zip(*n)
ax.scatter(left, right, bottom, color='purple', marker='o', label="Data Points")

# Adding labels and title
ax.set_tlabel("Omega")
ax.set_llabel("Alpha")
ax.set_rlabel("Gamma")
ax.set_title("Triangle plot")

# Show legend and plot
ax.legend()
plt.savefig("test5.png")
plt.close()

n = np.random.multivariate_normal(mean, covar, size=1000000)
n.shape
from corner import corner
corner(n)
plt.savefig("test6.png")
plt.close()
"""
#Testing out how autograd works
def multi_param_change(model, new_param_values, category):
    parameters = model.get_parameters()
    for i in range(len(category)):
        parameters["comps"]["cloud"][category[i]] = new_param_values[i]
    model.update_parameters(parameters)

def get_emission_grad(value, max, param_nr):
    #Can send in param_nr and value as int and float, respectively or as a list of ints and floats
    param =  ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "alpha", "beta", "gamma", "mu"]
    model = zodipy.Model(30 * u.micron)
    comm_tod = TODLoader("/mn/stornext/d5/data/metins/dirbe/data", "DIRBE")
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
    #parameters = model.get_parameters()
    if isinstance(param_nr, (int, float)):
        multi_param_change(model, [value], [param[int(param_nr)]])
    else:
        for i in range(len(param_nr)):
            multi_param_change(model, [value[i]], [param[int(param_nr[i])]])
    emission = model.evaluate(coords, nprocesses=1)[0:max] #size of 68828. Can decide the size using [0:1000]
    return emission

#Function to get the gradient of S_zodi
def get_zodigrad(value, max, param_nr):
    dS_zodidOmega = jax.jacfwd(get_emission_grad)(value, max, param_nr)
    return dS_zodidOmega

#Function to get the gradient of S_zodi using jacobian
def get_zodigrad_jacobian(value, max):
    dS_zodidOmega = jax.jacobian(get_emission_grad)(value, max)
    return dS_zodidOmega

#Testing zodigrad function 
param_nr = 8 #alpha: 6, gamma: 8
alpha = 1.337069670593028
Omega = 7.65795555409711
gamma = 0.9420617939335804
"""
for i in tqdm(range(1)):
    print(i)
    test = get_zodigrad(gamma, 68828, param_nr)
    print(f"test={test}")
    print(f"Shape of test:{np.shape(test)}")

with open(f"Images/Gradients_gamma.txt", 'w') as file:
            file.write(f"Gradients\n")
            for i in range(len(test)):
                file.write(f"{test[i]}\n")
plt.plot(test)
plt.xlabel("x[*]")
plt.ylabel("y[*]")
plt.title("Gradient plot")
plt.savefig("Images/Gradient_plot_gamma.png")
plt.close()
"""
"""
#Testing difference between jacfwd and jacobian
test = get_zodigrad(77.65795555409711, 1000)
test_jacobian = get_zodigrad_jacobian(77.65795555409711, 1000)
difference_array = np.zeros(len(test))
for i in range(len(test)):
    difference_array[i] = np.abs(test_jacobian[i] - test[i])
plt.plot(difference_array)
plt.xlabel("x[*]")
plt.ylabel("y[*]")
plt.title("difference")
plt.savefig("Images/Gradient_difference.png")
plt.close()
#print(f"test={test}")
#print(f"Shape of test:{np.shape(test)}")
"""
"""
#Check if analytical is the same
#Stupid way
max = 68828
h = 0.00001
derivative = (get_emission_grad(gamma + h, max, param_nr).value - get_emission_grad(gamma - h, max, param_nr).value)/(2*h)
difference_array = np.zeros(len(test))
for i in range(len(test)):
    difference_array[i] = np.abs(derivative[i] - test[i])
plt.plot(difference_array)
plt.xlabel("x[*]")
plt.ylabel("y[*]")
plt.title("difference using Jax and stupid way (finite difference)")
plt.savefig("Images/Gradient_difference_gamma_00001.png")
plt.close()
#Relative difference 
rel_difference_array = np.zeros(len(test))
for i in range(len(test)):
    rel_difference_array[i] = (test[i] - derivative[i])/derivative[i]
plt.plot(rel_difference_array)
plt.xlabel("x[*]")
plt.ylabel("y[*]")
plt.title("relative difference using Jax and stupid way (finite difference)")
plt.savefig("Images/Gradient_relative_difference_gamma_00001.png")
plt.close()
"""
#Getting CHI squared 
def get_chi2(value, max, param_nr, noisy_data):
    chi2 = 0
    sigma = 0.1
    emission = get_emission_grad(value, max, param_nr)
    """
    #271.36474609375
    #270.3493068598618
    #Max absolute difference: 50.41745607
    #Max relative difference: 0.11771947
    #x: array(478.70224, dtype=float32)
    #y: array(428.284784)
    for i in tqdm(range(len(emission))):
            #Need to add a if-test to check if a Tracer wass sent through. It doesn't work with ".value"
            if type(emission[i]) == jax.interpreters.ad.JVPTracer:
                chi2 += ((noisy_data[i].value - emission[i])/sigma)**2
            else:
                chi2 += ((noisy_data[i].value - emission[i].value)/sigma)**2
    """
    #"""
    # 271.365478515625
    # 270.3125 base , 271.09375 with float 32
    #Max absolute difference: 49.015564
    #Max relative difference: 0.11407258
    #x: array(478.70306, dtype=float32)
    #y: array(429.6875, dtype=float32)
    if type(emission) == jax.interpreters.ad.JVPTracer:
        chi2 = jnp.sum(((noisy_data.value - emission)/sigma)**2)
    else:
        chi2 = jnp.sum(((noisy_data.value - emission.value)/sigma)**2)
    #"""
    return chi2

def get_chi2_gradient(parameter_value, max, param_nr, noisy_data):
    #dchi2_dOmega = jax.grad(get_chi2)(parameter_value, max, param_nr, noisy_data)
    dchi2_dOmega = jnp.array(jax.jacfwd(get_chi2)(parameter_value, max, param_nr, noisy_data))
    return dchi2_dOmega

def noise_maker(data, mu, sigma): 
    noisy_data = jnp.zeros(len(data))*data.unit
    for i in range(len(data)):
        n = np.random.normal(mu, sigma)*data.unit
        noisy_data[i] = data[i] + n
    return noisy_data
"""
x_0 = 0
y_0 = 1
z_0 = 2
i = 3
Omega = 4
n_0 = 5
alpha = 6
beta = 7
gamma = 8
mu = 9
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
parameter_value = jnp.array(77.65795555409711, dtype=jnp.float32)
max = 68828
param_nr = 4
model = zodipy.Model(30 * u.micron)
emission_og = get_emission_grad(parameter_value, max, param_nr)
noisy_data = noise_maker(emission_og[0:max], 0, 0.1)
#Testing Gradients
"""
testing_chi2 = get_chi2_gradient(parameter_value, max, param_nr, noisy_data)
print(f"Chi2 Gradient = {testing_chi2}")
#Printing to txt-file
with open(f"Images/Gradients_Chi2.txt", 'w') as file:
            file.write(f"Gradients\n")
            file.write(f"{testing_chi2}\n")
#Finding Chi2 the stupid way finite differences 2 ways:
#1.
h = 0.01
Chi_1 = get_chi2(parameter_value + h, max, param_nr, noisy_data)
Chi_2 = get_chi2(parameter_value - h, max, param_nr, noisy_data)
chi2gradient = (Chi_1 - Chi_2)/(2*h)
print(f"Chi2 gradient finite differences {chi2gradient}")
rel_diff = (chi2gradient - testing_chi2)/testing_chi2
# h = 0.01 gives us relative difference: -0.0010013377759605646
# h = 0.001 gives us relative difference: -0.006759255193173885
# h = 0.0001 gives us relative difference: 0.007635538000613451
print(f"Relative difference between grad and finite difference: {rel_diff}")
#2. 
get_chi2_partial = partial(get_chi2, max = max, param_nr = param_nr, noisy_data = noisy_data)
chi2gradient_jax = check_grads(get_chi2_partial, (parameter_value,), order=1, atol = 0.012, rtol = 0.012, eps=0.01)
#chi2gradient_jax = check_grads(get_chi2_partial, (parameter_value,), order=1, eps=0.01)
print(f"Chi2 gradient JAX {chi2gradient_jax}")
#"""
"""
plt.plot(testing_chi2)
plt.xlabel("x[*]")
plt.ylabel("y[*]")
plt.title("Gradient plot of Chi2")
plt.savefig("Images/Gradient_plot_Chi2.png")
plt.close()
"""

#Checking for all parameters
def Chi2_all_para_grad(model, noisy_data):
    param =  ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "alpha", "beta", "gamma", "mu"]
    param_arr = jnp.zeros(len(param))
    parameters = model.get_parameters()
    chi2_arr = jnp.zeros(len(param))
    for i in range(len(param)):
        param_arr = param_arr.at[i].set(parameters.get("comps").get("cloud").get(param[i]))
        #print(f"param_arr[{i}] = {param_arr[i]}")
    for i in tqdm(range(len(param))):
        Chi2_val = get_chi2_gradient(param_arr[i], max, i, noisy_data)
        print(f"{param[i]} has Chi2 = {Chi2_val}")
        """#Testing if gradient is correct
        h = 0.0001
        Chi_1 = get_chi2(param_arr[i] + h, max, i, noisy_data)
        Chi_2 = get_chi2(param_arr[i] - h, max, i, noisy_data)
        chi2gradient = (Chi_1 - Chi_2)/(2*h)
        print(f"Chi2 gradient finite differences {chi2gradient}")
        rel_diff = jnp.abs((chi2gradient - Chi2_val))/Chi2_val
        print(f"Relative difference between grad and finite difference: {rel_diff}")
        get_chi2_partial = partial(get_chi2, max = max, param_nr = i, noisy_data = noisy_data)
        chi2gradient_jax = check_grads(get_chi2_partial, (param_arr[i],), order=1, atol = 0.01, rtol = 0.01, eps=0.0001)
        #chi2gradient_jax = check_grads(get_chi2_partial, (parameter_value,), order=1, eps=0.01)
        print(f"Chi2 gradient JAX {chi2gradient_jax}")
        #"""
        chi2_arr = chi2_arr.at[i].set(Chi2_val)
    return chi2_arr
model = zodipy.Model(30 * u.micron)
Omega = 77.65795555409711 
Omega_idx = 4
#emission_og = get_emission_grad(Omega, max, Omega_idx)
#noisy_data = noise_maker(emission_og, 0.1, 0)

#chi2_arr = Chi2_all_para_grad(model, noisy_data)


#Implement scipy.optimize.minimize
#Need to make a function that returns the value and the gradient
def Chi2_and_Chi2Grad(value):
    max = 68828
    param_nr = 4
    emission_og = get_emission_grad(Omega, max, param_nr)
    noisy_data = noise_maker(emission_og, 0, 0.1)
    Chi2 = get_chi2(value, max, param_nr, noisy_data)
    Chi2_grad = get_chi2_gradient(value, max, param_nr, noisy_data)
    #print(f"Chi2 = {Chi2} and the gradient of Chi2 = {Chi2_grad}")
    return Chi2, Chi2_grad


#Omega = 77.65795555409711 
Omega_initial = 100
#Minimize_result = minimize(Chi2_and_Chi2Grad, (Omega_initial), jac=True)
#print(f"Minimize result: {Minimize_result}")

#Hamiltonian monte carlo application following steps from https://bayesianbrad.github.io/posts/2019_hmc.html
"""
x_0 = 0
y_0 = 1
z_0 = 2
i = 3
Omega = 4
n_0 = 5
alpha = 6
beta = 7
gamma = 8
mu = 9
"""
#Need to send in an array of current_parameter with length 10 that has dimension (2, 10)
#basically current_parameter[0:1][0:10] 
#[0][0:10] = Parameter value 
def HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma_array):
    max = 68828 #Length of the emission dataset. 
    p = np.array([])
    mu = 0
    for i in range(len(parameterVal_paramNr[0])): 
        p = np.append(p, np.random.normal(mu, sigma_array[i]))

    #print(f"p beginning = {p}")
    parameter_arr = np.zeros_like(parameterVal_paramNr) 
    parameter_arr[1][:] = parameterVal_paramNr[1][:]
    #parameterVal_paramNr[0][i] is basically current_q while parameter is q
    parameter = parameterVal_paramNr[0][:] 
    current_p = p[:]
    #print(f"parameter sent into Chi2_grad: {parameter}")
    Chi2_grad = get_chi2_gradient(parameter, max, parameterVal_paramNr[1][:], noisy_data)  
    #print(f"Current Chi2_grad = {Chi2_grad}")
    p = p - epsilon*Chi2_grad/2
    #print(f"p = {p}")
    for j in range(1, steps, steps-1):
        parameter = parameter + epsilon*p
        if (j!=steps):
            Chi2_grad = get_chi2_gradient(parameter, max, parameterVal_paramNr[1][:], noisy_data)
            p = p - epsilon * Chi2_grad 

    Chi2_grad = get_chi2_gradient(parameter, max, parameterVal_paramNr[1][:], noisy_data)
    #print(f"Updated parameter = {parameter}")    
    p = p - epsilon*Chi2_grad/2

    p = -p 

    current_Chi2 = get_chi2(parameterVal_paramNr[0][:], max, parameterVal_paramNr[1][:], noisy_data)
    #print(f"current_Chi2 = {current_Chi2}")
    current_K = np.sum(current_p**2)/2
    #print(f"current_K = {current_K}")
    proposed_Chi2 = get_chi2(parameter, max, parameterVal_paramNr[1][:], noisy_data) 
    #print(f"proposed_Chi2 = {proposed_Chi2}")
    proposed_K = np.sum(p**2)/2
    #print(f"proposed_K = {proposed_K}")

    rng = np.random.default_rng()
    number = rng.random()
    probability = current_Chi2-proposed_Chi2+current_K-proposed_K
    #print(f"Probability = {probability}")
    #print(f"exp(Probability) = {np.exp(probability)}")
    #print(f"number = {number}")
    if number < np.exp(probability):
        #print(f"Accepted")
        #print(f"parameter before sent into parameter array = {parameter}")
        parameter_arr[0][:] = np.array(parameter, dtype='float64')
        parameter_arr = parameter_arr.astype(np.float64)
        #print(f"returned:{parameter_arr[0][:]}")
        return parameter_arr, 1, proposed_Chi2 #accepted
    else:
        #print(f"Rejected")
        #print(f"returned:{parameterVal_paramNr[0][:]}")
        return parameterVal_paramNr, 0,  current_Chi2 #rejected
#Code that tunes the stepsize
def tuning_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma):
    #Tuning to find good fitting epsilon
    epsilon_arr = [epsilon]
    accepted = 0
    new_parameterVal_paramNr, result, chi2 = HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma)
    #print(f"First {parameterVal_paramNr[0]}")
    #print(f"Second {new_parameterVal_paramNr[0]}")
    accepted += result
    iterations = 100 # will give 100 iterations
    for i in tqdm(range(iterations-1)):
        new_parameterVal_paramNr, result, chi2 = HMC(epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma)
        accepted += result

    acceptance_probability = accepted/iterations
    print(f"acceptance probability = {acceptance_probability}")
    AP_arr = [acceptance_probability]
    #0.4-0.9 are good targets
    lower_bound = 0.6
    upper_bound = 0.7
    while acceptance_probability < lower_bound or acceptance_probability > upper_bound:
        if acceptance_probability < lower_bound:
            epsilon = epsilon*0.9 #Reduce epsilon by 10%
        if acceptance_probability > upper_bound:
            epsilon = epsilon*1.1 #Increase epsilon by 10%
        print(f"Current epsilon: {epsilon}") 
        epsilon_arr = np.append(epsilon_arr, epsilon) #Put all the epsilons in a list 
        #Running again to see if the epsilon improved
        accepted = 0   
        new_parameterVal_paramNr, result, chi2 = HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma)
        accepted += result 
        for i in tqdm(range(iterations-1)):
            new_parameterVal_paramNr, result, chi2 = HMC(epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma)
            accepted += result               
        acceptance_probability = accepted/iterations
        AP_arr = np.append(AP_arr, acceptance_probability)
        print(f"acceptance probability = {acceptance_probability}")
    #Making txt files of epsilon and the acceptance rate
    with open(f"Images/HMC_epsilon_AP.txt", 'w') as file:
            file.write(f"Epsilon\tAcceptance Rate\n")
            for i in range(len(epsilon_arr)):
                file.write(f"{epsilon_arr[i]} \t {AP_arr[i]}\n")
    return epsilon
#Function to find the sigma
def find_sigma(parameterVal_paramNr, noisy_data, filename):
    max = 68828 #Length of the emission dataset. 
    #Find value that yields a change of 1 in Chi2
    limit = 1
    change = np.ones_like(parameterVal_paramNr[0])
    for i in tqdm(range(len(change))):
        difference = 1000
        parameter_1 = parameterVal_paramNr[0][i]
        Chi2_1 = get_chi2(parameter_1, max, int(parameterVal_paramNr[1][i]), noisy_data)
        while difference > limit:
            change[i] *= 0.1 
            parameter_2 = parameterVal_paramNr[0][i] + change[i]
            Chi2_2 = get_chi2(parameter_2, max, int(parameterVal_paramNr[1][i]), noisy_data)
            difference = np.mean(abs(Chi2_1 - Chi2_2))
            #print(f"From Change = {change[i]} gives difference = {difference}")
    #print(f"change array is = {change}")
    #print(f"1/change = {1/change}")
    with open(f"Images/{filename}_sigma.txt", "w") as file:
        file.write(f"{change}")
    file.close()
    return change
#Testing if tuning works

#"""
epsilon = 1.7338601276121614e-05 #stepsize want stepsize giving 0.4-0.9 acceptancerate 
#1.7338601276121614e-05 gave 0.88 acceptrate
#1.8750000000000002e-05 gave 0.7
#1.9031609322575876e-05 gave 0.69
#1.789e-05 gave (0.44) for 50 iterations 10 steps
#  1.79e-05 gave (0.996) for 2000 iterations 10 steps
# 1.7888e-05  gave (0.993) for 1000 iterations 10 steps
# 1.7899e-05 (0.85) 100 iterations 10 steps
#1.78999e-05 (0.956) 500 iterations 10 steps
# 1.85e-5 (0.0) for 100 iterations 10 steps
# 1.825-5 (0.01) for 100 iterations 10 steps
# 1.82-5 (0.00) for 100 iterations 10 steps
# 1.815 (0.016) for 100 iterations 10 steps
# 1.815 (0.958) for 1000 iterations 10 steps
# 1.825 (0.9815) for 2000 iterations 10 steps
# 1.85 (0.00) for 500 iterations 10 steps
# 1.835 (0.626) for 500 iterations 10 steps
# 1.835 (0.8475) for 2000 iterations 10 steps
# 1.84 (0.615) for 1000 iterations 10 steps
# 1.839e-05 (0.9385) for 2000 iterations 10 steps
# 1.84 (0.8683333333333333) for 3000 iterations 10 steps
# 1.845 (0.266) for 1500 iterations 10 steps
# 1.843 (0.6289473684210526) for 1900 iterations 10 steps
# 1.8431 (0.779) for 4000 iterations 10 steps
steps = 10 #Recommended 
Omega = 77.658  #77.65795555409711 
Omega_idx = 4
gamma = 0.942 #0.9420617939335804
gamma_idx = 8 
beta = 4.14#4.141500415758664
beta_idx = 7
alpha = 1.34#1.337069670593028
alpha_idx = 6

parameters = [Omega, gamma, alpha, beta]
parameters_actual = [Omega , gamma, alpha, beta]
#parameters = [alpha]
parameter_index = [Omega_idx, gamma_idx, alpha_idx, beta_idx]
#parameter_index = [alpha_idx]
parameterVal_paramNr = jnp.array([parameters, parameter_index], dtype=jnp.float32)
emission_og = get_emission_grad(parameters_actual, max, parameter_index) #BE CAREFUL YOU NEED TO MAKE SURE YOU HAVE tHE RIGHT EMISSION
noisy_data = noise_maker(emission_og, 0, 0.1) #(emission, mu, sigma)
#noisy_data = noise_maker(emission_og, 0, 0) #TEST noiseless data should gi
#sigma = find_sigma(parameterVal_paramNr, noisy_data)
#good_fit_epsilon = tuning_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma)
#print(f"Good fit epsilon is: {good_fit_epsilon}")

#Function applies HMC and returns an array of the parameters
def get_parameter_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, iterations, sigma, filename):
    #good_fit_epsilon = tuning_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma)
    good_fit_epsilon = 1.8433e-05 #Using this for bug testing
    accepted = 0
    #Initial parameter values
    new_parameterVal_paramNr, result, chi2 = HMC(epsilon, steps, parameterVal_paramNr, noisy_data, sigma)
    accepted += result
    #print(f"PARAMETERS = {new_parameterVal_paramNr[0][:]}")
    parameter_value_arr = np.array([new_parameterVal_paramNr[0][:]], dtype='float64')
    chi2_array = np.array(chi2, dtype="float64")
    #print(f"PARAMETERS VALUE ARRAY : {parameter_value_arr}")
    #np.shape() = (amount of parameters, iterations)
    for i in tqdm(range(iterations-1)):
        new_parameterVal_paramNr, result, chi2 = HMC(good_fit_epsilon, steps, new_parameterVal_paramNr, noisy_data, sigma)
        #print(f"PARAMETERS = {new_parameterVal_paramNr[0][:]}")
        accepted += result
        parameter_value_arr = np.append(parameter_value_arr, np.array([new_parameterVal_paramNr[0][:]]), axis=0)
        chi2_array = np.append(chi2_array, chi2)
    #print(f"PARAMETERS VALUE ARRAY UPDATED: {parameter_value_arr}")
    print(f"Acceptance rate: {accepted/iterations}")
    #Write to a file
    with open(f"Images/{filename}_accept_rate.txt", "w") as file:
        file.write(f"Epsilon: {good_fit_epsilon}")
        file.write(f"Acceptance rate: {accepted/iterations}")
    file.close()
    param =  ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "alpha", "beta", "gamma", "mu"]
    with open(f"Images/{filename}.txt", 'w') as file:
            for i in range(len(new_parameterVal_paramNr[1][:])):
                #print(f"{param[int(parameterVal_paramNr[1][i])]}")
                file.write(f"{param[int(new_parameterVal_paramNr[1][i])]}\t")
            file.write(f"chi2\t")
            file.write(f"\n")
            for i in range(iterations):
                for j in range(len(new_parameterVal_paramNr[1][:])):
                    #print(f"What was written to the file: {parameter_value_arr[i][j]}")
                    file.write(f"{parameter_value_arr[i][j]}\t")
                file.write(f"{chi2_array[i]}\t")
                file.write(f"\n")
    file.close()
    param_name = np.array([], dtype=object)
    for i in range(len(new_parameterVal_paramNr[1][:])):
        param_name = np.append(param_name, param[int(new_parameterVal_paramNr[1][i])])
    #print(f"param_name = {param_name}")
    return parameter_value_arr, param_name, chi2_array

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
    plt.savefig(f"Images/HMC_corner_{iterations}_iterations_{len(parameter_name)}_parameters.png")
    plt.close()
    #Plotting Chi2 against iterations
    plt.plot(Chi2_array)
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.savefig(f"Images/HMC_corner_{iterations}_iterations_Chi2.png")
    plt.close()

#Corner plot using txt files
def get_corner_plt_readfile(filename):
    with open(filename, 'r') as file:
        labels = file.readline().strip().split()
    data = np.loadtxt(filename, skiprows = 1)
    iterations = len(data[:,-1])

    #Plotting Chi2 against iterations unfiltered
    plt.plot(data[:,-1])
    plt.title(f"Chi2 unfiltered")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.savefig(f"Images/HMC_corner_unfiltered_{iterations}_iterations_Chi2.png")
    plt.close()

    #filter data
    threshold = np.mean(data[:,-1])
    filter = data[:, -1] < threshold
    data = data[filter]

    result = np.vstack(data)
    figure = corner.corner(result, labels = labels)
    plt.title(f"{len(data[0][:])} data points")
    plt.savefig(f"Images/HMC_corner_{iterations}_iterations_{len(data[:][0])-1}_parameters.png")
    plt.close()
    #Plotting Chi2 against iterations
    plt.plot(data[:,-1])
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.savefig(f"Images/HMC_corner_{iterations}_iterations_Chi2.png")
    plt.close()

#
#Testing out the code
epsilon = 0.001

steps = 10 #IMPORTANT TO HAVE 10 STEPS AS THAT'S WHAT I USED DURING TUNING

iterations = 4000 #Important, need enough iterations so the values have some variance for corner.

parameters = [Omega, gamma, alpha, beta]
parameterVal_paramNr = jnp.array([parameters, parameter_index], dtype=jnp.float32)
parameter_index = [Omega_idx, gamma_idx, alpha_idx, beta_idx]
emission_og = get_emission_grad(parameters, max, parameter_index) #BE CAREFUL YOU NEED TO MAKE SURE YOU HAVE tHE RIGHT EMISSION
noisy_data = noise_maker(emission_og, 0, 0.1) #(emission, mu, sigma)



filename = f"HMC_test_{iterations}_iterations"

#sigma = find_sigma(parameterVal_paramNr, noisy_data, filename)
sigma = [9.999999e-04, 9.999999e-09, 9.999999e-07, 9.999999e-07]
#parameter_value_arr, parameter_name, Chi2_array = get_parameter_HMC(epsilon, steps, parameterVal_paramNr, noisy_data, iterations, sigma, filename)

#get_corner_plt(parameter_value_arr, Chi2_array, parameter_name, iterations)
"""
parameters = model.get_parameters()
top = list(parameters.keys())
print(f"Keys:{top}")

middle = list(parameters["comps"].keys())
print(f"Middle Keys = {middle}")

bottom = list(parameters["comps"]["cloud"].keys())
print(f"Bottom Keys = {bottom}")

bottom_values = parameters["comps"]["cloud"]["x_0"]
print(f"Bottom values = {bottom_values}")
"""
#JAX doesn't like having text sent in so have to make a function that converts string into numbers and numbers into string
#String_list shoudl look like this 
# EX.1 ["comps", "cloud", ["Omega", "beta"]]
# Ex.2 ["comps", "cloud"] This will change all the parameters within parameter["comps"]["cloud"]
def parameter_list_str_to_nmbr(string_list, model):
    parameters = model.get_parameters()
    top = list(parameters.keys())
    #middle = list(parameters["comps"].keys())
    #bottom = list(parameters["comps"]["cloud"].keys())
    if string_list[0] in top:
        top_index = top.index(string_list[0])
        middle = list(parameters[top[top_index]].keys())
        if string_list[1] in middle:
            middle_index = middle.index(string_list[1])
            if len(string_list) > 2:
                bottom = list(parameters[top[top_index]][middle[middle_index]])
                bottom_arr = np.zeros(len(string_list[2]))
                for i in range(len(bottom_arr)):
                    if string_list[2][i] in bottom:
                        bottom_index = bottom.index(string_list[2][i])
                        bottom_arr[i] = int(bottom_index)
                    else:
                        print(f"Did not find {string_list[2][i]}")
            else:
                bottom = list(parameters[top[top_index]][middle[middle_index]])
                bottom_arr = np.arange(0, len(bottom))
        else:
            print(f"Did not find {string_list[1]}")
    else: 
        print(f"Did not find {string_list[0]}")

    return [top_index, middle_index, bottom_arr]

#Function that transalates parameter numbers into strings
def parameter_nmbr_to_str_list(number_array, model):
    parameters = model.get_parameters()
    #Top
    top = list(parameters.keys())
    top_str = top[number_array[0]]
    #Middle
    middle = list(parameters[top[number_array[0]]].keys())
    middle_str = middle[number_array[1]]
    #Bottom
    bottom = list(parameters[top[number_array[0]]][middle[number_array[1]]].keys())
    print(f"length of number array = {len(number_array)}")
    if len(number_array) > 2:
        bottom_str_arr = []
        for i in range(len(number_array[2])):
            bottom_str_arr = np.append(bottom_str_arr, bottom[int(number_array[2][i])])
    else:
        bottom_str_arr = []
        for i in range(len(bottom)):
            bottom_str_arr = np.append(bottom_str_arr, bottom[i])
    return [top_str, middle_str, bottom_str_arr]


"""
#Small test
string_list = ["comps", "cloud"]
print(f"String list: {string_list}")
number_array = parameter_list_str_to_nmbr(string_list, model)
print(f"Number array: {number_array}")
string_list_recovered = parameter_nmbr_to_str_list(number_array, model)
print(f"Recovered string list: {string_list_recovered}")
print(f"Trying to get a thing out from recovered list {string_list_recovered[2][0]}")
"""


#Timing the code
t1 = time.time()
total = t1-t0
print(f"The code took {total:.2f}s to run")