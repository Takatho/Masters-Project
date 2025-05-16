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
t0 = time.time()

#Importing altered Zodipy
import sys
sys.path.insert(0, "/uio/hume/student-u59/taabeled/Desktop/Masters thesis/Masters/Zodipy_Thomas/zodipy_Thomas/")
import zodipy
from zodipy import model_registry
#import zodipy

# Initialize a zodiacal light model at a wavelength/frequency or over a bandpass
model = zodipy.Model(30 * u.micron)
#Original Parameter
param = model.get_parameters()
#Overhead group
#print(param)

#Sub group, example: .get("comps")
#print(param.get("comps"))

#Parameter values, example: .get("cloud")
#print(param.get("comps").get("cloud"))

#Specified paramter value, example: .get("Omega")
#print(param.get("comps").get("cloud").get("Omega"))
"""
#Original Omega value = 77.65795555409711
Omega_val = param.get("comps").get("cloud").get("Omega")
print(f"Omega = {Omega_val}")

#Change specific Parameter 
param["comps"]["cloud"]["Omega"] = 70
#Update model with the new parameter
update_param = model.update_parameters(param)
Omega_val_update = param.get("comps").get("cloud").get("Omega")
#New Omega val = 70
print(f"Updated Omega = {Omega_val_update}")
#"""

#Function to change parameter
def param_change(model, new_param_val, category_1, category_2, category_3): 
    parameters = model.get_parameters()
    #parameter_val = parameters.get(category_1).get(category_2).get(category_3)
    #print(f"OG val: {parameter_val}")
    parameters[category_1][category_2][category_3] = new_param_val
    #print(f" New val: {parameter_val}")

    model.update_parameters(parameters)
    #new_parameters = updated_model.get_parameters()    
    return model

#Function to make and array of the value,  between 0 - 2*value
def param_array(model, category_1, category_2, category_3):
    parameters = model.get_parameters()
    value = parameters[category_1][category_2][category_3]
    array = np.linspace(0, 2*value, 5)
    return array, value

#Function that plots out ecliptic where we vary a specified parameter
def plotting_param_vary(model, new_param_val, category_1, category_2, category_3):
    if new_param_val == False:
        new_model = model
    else:
        new_model = param_change(model, new_param_val, category_1, category_2, category_3)

    array, original_value = param_array(new_model, category_1, category_2, category_3)

    lats = np.linspace(-90, 90, 100) * u.deg
    lons = np.zeros_like(lats)
    t0 = Time("2022-06-14")
    dt = TimeDelta(1, format="sec")
    obstimes = t0 + np.arange(lats.size) * dt

    coords = SkyCoord(
        lons,
        lats,
        frame=BarycentricMeanEcliptic, #Need to put in Galactic coordinates?
        obstime=obstimes,
    )
    Figure = plt.figure()
    plt.ylim(0, 60)
    plt.xlim(-100, 100)

    lines_plotted = plt.plot([]) 
    line_plotted = lines_plotted[0]
    # function takes frame as an input
    def AnimationFunction(frame): 
        # setting y according to frame
        # number and + x. It's logic
        param_change(new_model, array[frame], category_1, category_2, category_3)
        emission = new_model.evaluate(coords)
        y = emission 
        x = lats
        # line is set with new values of x and y
        line_plotted.set_data((x, y))
        plt.title(f"Parameter:{category_3} (Current value:{array[frame]:.3f})")
    anim = FuncAnimation(Figure, AnimationFunction, repeat = True,  frames=len(array)-1, interval=1)
    plt.xlabel("Latitude [deg]")
    plt.ylabel("Emission [MJy/sr]")
    anim.save(f"GIFS/Param_{category_3}.gif", dpi = 300)
    plt.close()

#Testing gif
#plotting_param_vary(model, new_param_val, category_1, category_2, category_3)
#Exmple of use: plotting_param_vary(model, False, "comps", "cloud", "Omega")
#In cloud we have parameters: x_0, y_0, z_0, i, Omega, n_0, alpha, beta, gamma, mu
#"""
cloud_params = ["x_0", "y_0", "z_0", "i", "Omega", "n_0", "alpha", "beta", "gamma", "mu"]
#for i in range(len(cloud_params)):
    #plotting_param_vary(model, False, "comps", "cloud", cloud_params[i])
#"""
#"""
def plotting_param_vary_in_Healpy(model, new_param_val, category_1, category_2, category_3):
    if new_param_val == False:
        new_model = model
    else:
        new_model = param_change(model, new_param_val, category_1, category_2, category_3)

    array, original_value = param_array(new_model, category_1, category_2, category_3)

    nside = 256
    pixels = np.arange(hp.nside2npix(nside))
    #print(f"pixels = {pixels}")
    lon, lat = hp.pix2ang(nside, pixels, lonlat=True)
    skycoord = SkyCoord(lon, 
                        lat, 
                        unit=u.deg, 
                        frame="galactic", 
                        obstime=Time("2022-01-14"))
    
    for i in range(len(array)):
        param_change(model, array[i], category_1, category_2, category_3)
        emission = model.evaluate(skycoord)
        hp.mollview(emission,unit="MJy/sr",cmap="afmhot",
                    min=0,max=80,title=f"Zodiacal light at 30 µm (2022-01-14) {category_3}:{array[i]:.2f}",)
        plt.savefig(f"GIFS_{category_3}/Healpy_{category_3}_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()
"""
#Healpy gifs
for i in range(len(cloud_params)):
    model = zodipy.Model(30 * u.micron) #Return to original model.
    plotting_param_vary_in_Healpy(model, False, "comps", "cloud", cloud_params[i])
    print(f"Images for {cloud_params[i]} completed")
#"""

#Plot the cross section of the number density.
#Function to plot the number density
#"""
def number_density(Plane):
    #"""
    N = 200
    x = np.linspace(-5, 5, N) * u.AU  # x-plane
    y = np.linspace(-5, 5, N) * u.AU  # y-plane
    z = np.linspace(-2, 2, N) * u.AU  # z-plane
    Figure = plt.figure()
    # function takes frame as an input
    def AnimationFunction(frame): 
        #full rotation
        density_grid = zodipy.grid_number_density(x, y, z, 
        obstime=Time("2021-01-01T00:00:00", scale="utc"), model="DIRBE",)
        #density_grid has shape (6, 200, 200, 200)
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
        plt.title(f"Cross section of the number density in the DIRBE model frame:{frame}, {Plane}-plane")
        
    anim = FuncAnimation(Figure, AnimationFunction, repeat = True,  frames=10, interval=1)
    anim.save(f"Images/number_density_{Plane}.gif", dpi = 300)
    plt.show()
    print("GIF made")
    #"""
    #print(f"Function done")
#number_density, Need plane inserted xy, xz, yz
#number_density("xy") 
#number_density("xz") 
#number_density("yz") 
#"""
#Rotating number density
def number_density_rotation(Plane):
    model = zodipy.Model(30 * u.micron, extrapolate=True)
    param = model.get_parameters()
    N = 200
    x = np.linspace(-5, 5, N) * u.AU  # x-plane
    y = np.linspace(-5, 5, N) * u.AU  # y-plane
    z = np.linspace(-2, 2, N) * u.AU  # z-plane
    Figure = plt.figure()
    # function takes frame as an input
    def AnimationFunction(frame): 
        print(f"{Plane}-Frame:{frame}")
        #Change parameter that rotates the cloud
        #Original Omega value = 77.65795555409711
        OG_omega = param.get("comps").get("cloud").get("Omega")
        new_model = param_change(model, OG_omega+frame*36, "comps", "cloud", "Omega") #frame*36 spins 360 degres
        #Need to inject this new model into grid_number_density part        
        #Changing model parameters.
        #print(type(model_registry.get_model("DIRBE"))) #<class 'zodipy.zodiacal_light_model.Kelsall'>
        #print(type(new_model)) #<class 'zodipy.model.Model'>
        density_grid = zodipy.grid_number_density(x, y, z, #density_grid has shape (6, 200, 200, 200)
        obstime=Time("2021-01-01T00:00:00", scale="utc"), model=new_model._ipd_model,)
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
        plt.title(f"Cross section of the number density in the DIRBE model frame:{frame}, {Plane}-plane")
        
    anim = FuncAnimation(Figure, AnimationFunction, repeat = True,  frames=10, interval=1)
    anim.save(f"Images/number_density_{Plane}_spinning.gif", dpi = 300)
    plt.show()
    print("Spinny GIF made")
    plt.close()

#number_density_rotation("yz")
#number_density_rotation("xy")
#number_density_rotation("xz")

#print(f"Rotating number density Plotted")
#DIRBE TOD (Time-ordered data)
def Dirbe_TOD(model, new_param_val, category_1, category_2, category_3): 
    comm_tod = TODLoader("/mn/stornext/d23/cmbco/dirbe/data", "DIRBE")
    comm_tod.init_file('05_nside512_V23', '')
    pix = comm_tod.load_field(f'000185/05_A/pix')
    nside = 512 #hp.npix2nside(pix)
    #print(f"nside = {nside}")

    #pix = np.arange(hp.nside2npix(nside)) #Testing if earlier pix causes problems
    #print(f"Pix from arrange: {np.shape(pix)}")
    lon, lat = hp.pix2ang(nside, pix, lonlat=True) 
    #print(f"lon: {np.shape(lon)}")
    #print(f"lat: {np.shape(lat)}")
    t0 = Time("2022-01-14")
    dt = TimeDelta(1, format="sec")
    obstimes = t0 + np.arange(lat.size) * dt
    Observations = np.linspace(0, len(obstimes), len(obstimes))
    coords = SkyCoord(
            lon,
            lat, 
            unit=u.deg, 
            frame="galactic", #Need to put in Galactic coordinates?
            obstime=obstimes #Need array of obstimes 
        )
    value_OG = param.get(category_1).get(category_2).get(category_3) 
    emission = model.evaluate(coords)
    plt.plot(Observations[0:10000], emission[0:10000], color=plt.cm.viridis(0), label=f"Original {category_3} = {value_OG:.2f}")
    if type(new_param_val) == int or float:     
        param_change(model, new_param_val, category_1, category_2, category_3)
    #Bug fix
    nside = 256
    pixels = np.arange(hp.nside2npix(nside))
    lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

    coords = SkyCoord(lon, lat, unit=u.deg, frame="galactic", obstime=Time("2022-01-14"))
    #Bug fix test
    emission = model.evaluate(coords)
    #Observations with new 
    plt.plot(Observations[0:10000], emission[0:10000], color=plt.cm.viridis(1/2), linestyle=":" ,label=f"{category_3} = {new_param_val:.2f}")
    plt.xlabel("Observations")
    plt.ylabel("Emission [MJy/sr]")
    plt.legend()
    plt.savefig(f"Images/ecliptic_scan_DIRBE_TOD_Obs_{category_3}_{new_param_val:.2f}.png", dpi=300, bbox_inches="tight")
    plt.close()
    """
    #Latitude
    plt.plot(lat, emission, linestyle=":")
    plt.xlabel("Latitude [deg]")
    plt.ylabel("Emission [MJy/sr]")
    plt.savefig("Images/ecliptic_scan_DIRBE_TOD_Lat.png", dpi=300, bbox_inches="tight")
    plt.close()
    #Longtitude
    plt.plot(lon, emission, linestyle=":")
    plt.xlabel("Longtitude [deg]")
    plt.ylabel("Emission [MJy/sr]")
    plt.savefig("Images/ecliptic_scan_DIRBE_TOD_Long.png", dpi=300, bbox_inches="tight")
    plt.close()
    """
    #"""
    print(f"Emission shape = {emission.shape}")
    print(f"pix shape = {pix.shape}")
    print(f"Need {12*nside**2} data")
    hp.mollview(emission,unit="MJy/sr",cmap="afmhot",
                    min=0,max=80,title=f"Zodiacal light at 30 µm (2022-01-14) with Dirbe TOD",)
    plt.savefig(f"Images/A_healpy_Dirbe_TOD.png", dpi=300, bbox_inches="tight")
    plt.close()
    #"""
#Original Omega value = 77.65795555409711
#model = zodipy.Model(30 * u.micron)
Omega_val = param.get("comps").get("cloud").get("Omega")
#Dirbe_TOD(model, Omega_val+90, "comps", "cloud", "Omega")
#print("Dirbe TOD plotted")


#IMPLEMENTING JAX AND MAKING A NOISY DATA SET AS A SANITY CHECK

#Function to change parameters
def par_change(model, new_param_val, category_1, category_2, category_3): 
    parameters = model.get_parameters()
    parameters[category_1][category_2][category_3] = new_param_val
    model.update_parameters(parameters)

#Function that adds noise to data takes: 23s for 688283, 1s for 10000
#data: from Zodi
#sigma: standard deviation
def noise_maker(data, sigma, mu): 
    noisy_data = jnp.zeros(len(data))*data.unit
    for i in range(len(data)):
        n = np.random.normal(mu, sigma)*data.unit
        noisy_data[i] = data[i] + n
    return noisy_data

#Function that gives emission from TOD :38s
def get_emission(model):
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
    emission = model.evaluate(coords[0:68828:10], nprocesses=1) #~35.45s to run. 8.4s with 6 cores
    #time1 = time.time()
    #print(f"{time1-time0}s")
    Observations = jnp.linspace(0, len(obstimes), len(obstimes))
    return emission, Observations

#Function for plotting of emission: 2s
def emission_plot(emission, emission_noise, observation, size, savefig_name):
    plt.plot(observation[0:size], emission_noise[0:size], color=plt.cm.viridis(1/2), linestyle="-" ,label=f"Noisy emission")
    plt.plot(observation[0:size], emission[0:size], color=plt.cm.viridis(2/2), linestyle="-" ,label=f"Fitted emission")
    plt.xlabel("Observations")
    plt.ylabel("Emission [MJy/sr]")
    plt.legend()
    plt.title("DIRBE_TOD with noisy emission")
    plt.savefig(savefig_name, dpi=300, bbox_inches="tight")
    plt.close()

model = zodipy.Model(30 * u.micron)
#emission_og, observation_og = get_emission(model)
#noisy_emission = noise_maker(emission_og[0:10000], 0.1, 0)
#emission_plot(emission_og, noisy_emission, observation_og, 10000, "Images/noisy_ecliptic_scan_DIRBE_TOD_Obs.png")

#Function for finding the chi
#data: Noisy arbitrary data
#model: model of IPD
#sigma: standard deviation
def get_chi2(data, model, sigma, with_jax=False):
    chi2 = 0
    emission, observations = get_emission(model)
    if with_jax==False:
        chi2 = jnp.sum(((data.value - emission.value)/sigma)**2)
    else:
        chi2 = 10
        #for i in range(len(data)):   
    return chi2

#Function to get the gradient of S_zodi
def get_zodigrad(category_1, category_2, category_3, model):
    
    emission, observations = get_emission(model)
    dS_zodidOmega = jax.grad(emission())
    return dS_zodidOmega 

#Function for finding the best fit parameter
def Best_fit_parameter(data, sigma, category_1, category_2, category_3, initial_guess, size, MHS=False, with_jax=False):
    parameter = jnp.linspace(initial_guess/2, initial_guess*2, 10) #Takes a long time with 1000 points, 18min with 100
    smallest_chi2 = 1e10
    Best_fit_parameter = initial_guess
    Best_fit_emission = data
    model = zodipy.Model(30 * u.micron)
    #Timing
    t0 = time.time()
    chi_arr = np.zeros(len(parameter))
    #Find Smallest Chi2 either through Metropolis Hasting Sampling or 
    parameter_val = initial_guess
    if MHS == True:
        parameter_val = parameter_val
        for i in range(len(parameter)):
            n = np.random.normal(0, sigma)
            par_change(model, parameter_val+n, category_1, category_2, category_3)
            emission_1, observations = get_emission(model)
            chi2_0 = get_chi2(data=data[0:size],model=model,sigma=sigma,with_jax=with_jax)
            if chi2_0 < smallest_chi2:
                smallest_chi2 = chi2_0
                Best_fit_parameter = parameter[i]
                Best_fit_emission = emission_1
                chi_arr[i] = smallest_chi2
                #Timing
                t1 = time.time()
                print(f"i:{i} took {t1-t0}s")
            else:
                probability = (-(chi2_0 - smallest_chi2)/2)
                print(f"Probability: {probability}")
                rng = np.random.default_rng()
                if np.log(rng.random()) < probability:
                    smallest_chi2 = chi2_0
                    Best_fit_parameter = parameter[i]
                    Best_fit_emission = emission_1
                    chi_arr[i] = smallest_chi2
                    #Timing
                    t1 = time.time()
                    print(f"i:{i} took {t1-t0}s")
                else:
                    smallest_chi2 = smallest_chi2
                    chi_arr[i] = smallest_chi2
            print(f"Chi2 = {chi_arr[i]}")
            print(f"Parameter = {parameter[i]}")
    #Finding smallest chi normally
    else:                
        for i in range(len(parameter)):
            par_change(model, parameter[i], category_1, category_2, category_3)
            emission, observations = get_emission(model)
            chi2 = get_chi2(data[0:size], model, sigma, False)
            if chi2 < smallest_chi2:
                smallest_chi2 = chi2
                Best_fit_parameter = parameter[i]
                Best_fit_emission = emission
            #Timing
            t1 = time.time()
            print(f"i:{i} took {t1-t0}s")

    emission_plot(Best_fit_emission, data, observations, size, f"Images/best_fit_{category_3}_ecliptic_scan_DIRBE_TOD_Obs.png")  
    print(f"The best fit parameter for {category_3}: {Best_fit_parameter}")  
    print(f"With chi2 = {smallest_chi2}")
    return Best_fit_parameter, smallest_chi2, Best_fit_emission, chi_arr

#BF_para, small_chi2, BF_emiss, chi_arr = Best_fit_parameter(noisy_emission, 0.1, "comps", "cloud", "Omega", 70, size=10000, MHS=True, with_jax=False)

#SANITY CHECK NB! note:put labels
def plot_Chi2(initial_val, category_1, category_2, category_3, noise_data, sigma, with_jax=False):
    model = zodipy.Model(30 * u.micron)
    param_arr = np.linspace(initial_val/2, initial_val*2, 10)
    chi2_arr = np.zeros_like(param_arr)
    for i in range(len(param_arr)):
        par_change(model, param_arr[i], category_1, category_2, category_3)
        chi2_arr[i] = get_chi2(noise_data, model, sigma, with_jax=False)
    plt.plot(param_arr, chi2_arr, label=f"Parameter:{category_3}")
    plt.legend()
    plt.show()
    plt.savefig(f"Images/Chi2_param.png", dpi=300, bbox_inches="tight")

#plot_Chi2(70, "comps", "cloud", "Omega", noisy_emission, 0.1, with_jax=False)

#MHS function
def get_MHS(noise_data, initial_val, size, category_1, category_2, category_3):

    model = zodipy.Model(30 * u.micron)
    param_arr = np.zeros(size)
    chi2_arr = np.zeros_like(param_arr)
    param_arr[0] = initial_val
    par_change(model, initial_val, category_1, category_2, category_3)
    chi2_arr[0] = get_chi2(noise_data, model, 0.1, with_jax=False)
    rng = np.random.default_rng()
    accept_rate = 0
    for i in tqdm(range(1, len(param_arr), 1)):
        n = np.random.normal(0, 0.1) #step size
        param_prop = param_arr[i-1] + n
        par_change(model, param_prop, category_1, category_2, category_3)
        chi2_prop = get_chi2(noise_data, model, 0.1, with_jax=False)
        probability = (-(chi2_prop - chi2_arr[i-1])/2) 
        if probability > np.log(rng.random()):
            chi2_arr[i] = chi2_prop
            param_arr[i] = param_prop
            accept_rate += 1
        else:
            chi2_arr[i] = chi2_arr[i-1]
            param_arr[i] = param_arr[i-1]
    
    print(f"{accept_rate} out of {size-1}")
    plt.plot(param_arr)
    plt.title(f"Plot of {category_3}")
    plt.savefig(f"Images/param_iter.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.plot(chi2_arr)
    plt.title(r"Plot of $\chi^2$")
    plt.savefig(f"Images/Chi2_iter.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.hist(param_arr)
    plt.title(f"Histogram for {category_3}")
    plt.savefig(f"Images/param_hist.png", dpi=300, bbox_inches="tight")
    #write values into a file
    with open("Images/param_chi_val.txt", 'w') as file:
        file.write(f"chi2\t{category_3}\n")
        for i in range(len(chi2_arr)):
            file.write(f"{chi2_arr[i]}\t{param_arr[i]}\n")
    return chi2_arr, param_arr

#chi2_arr, param_arr = get_MHS(noisy_emission, 77.66, 10,"comps", "cloud", "Omega")


#Function for changing multiple variables within the component and cloud section
#model: model = zodipy.Model(30 * u.micron)
#new_param_values: array of new parameter values
#category: array with same length as new_param_values with the parameters that we want to change
def multi_param_change(model, new_param_values, category):
    parameters = model.get_parameters()
    for i in range(len(category)):
        parameters["comps"]["cloud"][category[i]] = new_param_values[i]
    model.update_parameters(parameters)

#MHS function
def get_MHS_moreparams(noise_data, initial_values, step_size, size, category, name):
    model = zodipy.Model(30 * u.micron)
    param_arr = np.zeros((len(initial_values), size))
    param_prop = np.zeros(len(initial_values))
    chi2_arr = np.zeros(size)
    for i in range(len(initial_values)):
        param_arr[i][0] = initial_values[i]
    multi_param_change(model, initial_values, category)
    chi2_arr[0] = get_chi2(noise_data, model, 0.1, with_jax=False)
    rng = np.random.default_rng()
    accept_rate = 0
    #Get Covariance matrix
    Covariance_matrix = find_coavriance(noise_data, initial_values, step_size, category)
    mean = np.zeros(len(category))
    n = np.random.multivariate_normal(mean, Covariance_matrix, size=size)
    #Metropolis Hastings
    for i in tqdm(range(1, size, 1)):
        for j in range(len(initial_values)):
            #n = np.random.normal(0, step_size[j]) #step size
            param_prop[j] = param_arr[j][i-1] + n[i][j]

        multi_param_change(model, param_prop, category)
        chi2_prop = get_chi2(noise_data, model, 0.1, with_jax=False)
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
    with open(f"Images_5/accept_rate_{size}_{name}_{len(initial_values)}_params.txt", 'w') as file:
        file.write(f"accept rate: {accept_rate} out of {size-1}\n")
    for i in range(len(initial_values)):
        plt.plot(param_arr[i][:])
        plt.title(f"Plot of {category[i]}")
        plt.savefig(f"Images_5/param_{category[i]}_iter_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.hist(param_arr[i][:])
        plt.title(f"Histogram for {category[i]}")
        plt.savefig(f"Images_5/param_{category[i]}_hist_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
        plt.close()
    plt.plot(chi2_arr)
    plt.title(r"Plot of $\chi^2$")
    plt.savefig(f"Images_5/Chi2_iter_{size}_{name}_{len(initial_values)}_params.png", dpi=300, bbox_inches="tight")
    plt.close()
    #write values into a file
    for j in range(len(initial_values)):
        with open(f"Images_5/param_{category[j]}_chi_val_{size}_{name}_{len(initial_values)}_params.txt", 'w') as file:
            file.write(f"chi2\t{category[j]}\n")
            for i in range(len(chi2_arr)):
                file.write(f"{chi2_arr[i]}\t{param_arr[j][i]}\n")
    return chi2_arr, param_arr

#Covariance Function
def find_coavriance(noise_data, initial_values, step_size, category):
    size = 5000 #How many points normally 5000
    #Code from get_MHS_moreparams
    model = zodipy.Model(30 * u.micron)
    param_arr = np.zeros((len(initial_values), size))
    param_prop = np.zeros(len(initial_values))
    chi2_arr = np.zeros(size)
    for i in range(len(initial_values)):
        param_arr[i][0] = initial_values[i]
    multi_param_change(model, initial_values, category)
    chi2_arr[0] = get_chi2(noise_data, model, 0.1, with_jax=False)
    rng = np.random.default_rng()
    accept_rate = 0
    for i in tqdm(range(1, len(param_arr[0]), 1)):
        for j in range(len(initial_values)):
            n = np.random.normal(0, step_size[j]) #step size
            param_prop[j] = param_arr[j][i-1] + n
        multi_param_change(model, param_prop, category)
        chi2_prop = get_chi2(noise_data, model, 0.1, with_jax=False)
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
    if len(initial_values) > 2:
        covariance_matrix = np.cov(param_arr)
        print(f"Covariance matrix: {covariance_matrix}")
    else:
        covariance_matrix = np.array([[np.cov(param_arr)]])
        print(f"Covariance matrix: {covariance_matrix}")
    with open(f"Images_5/Covariance_matrix_{size}_{len(initial_values)}_params.txt", 'w') as file:
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
"""
init_val = [77.66, 1.34, 0.94]
param =  ["Omega", "alpha", "gamma"]
stepsize = [0.1, 0.001, 0.001]
"""
size = 5000 #100k
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
"""
init_val = [77.65795555409711]
param =  ["Omega"]
stepsize = [0.1]
#"""
#chi2_arr, param_arr = get_MHS_moreparams(noisy_emission, init_val, stepsize, size, param, name="covariance")


def plotting_3P(filename):
    with open(f"{filename}", "r") as f:
        label = f.readline().strip().split("\t")
    parameter_list = {"Omega":r"\Omega", "alpha":r"\alpha", "gamma":r"\gamma"}
    data = np.loadtxt(filename, skiprows=1, usecols=1)
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.ylabel(fr"${parameter_list[label[1]]}$")
    #plt.xlabel("")
    #plt.xlim(90000, len(data))
    plt.savefig(f"{filename}_iterations.png")
    plt.close()
filename_1 = "Images_6/param_alpha_chi_val.txt"
plotting_3P(filename_1)

filename_2 = "Images_6/param_gamma_chi_val.txt"
plotting_3P(filename_2)

filename_3 = "Images_6/param_Omega_chi_val.txt"
plotting_3P(filename_3)

#Timing the code
t1 = time.time()
total = t1-t0
print(f"The code took {total:.2f}s to run")
