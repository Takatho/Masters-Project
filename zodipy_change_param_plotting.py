import matplotlib.pyplot as plt
import numpy as np
import corner


#Function to get values from txt-file
def read_txt(txt_file):
    array_1, array_2 = np.loadtxt(txt_file,skiprows=1, unpack=True)
    return array_1, array_2
"""
chi2_arr, Omega_arr = read_txt("Images/param_Omega_chi_val.txt")
chi2_arr, alpha_arr = read_txt("Images/param_alpha_chi_val.txt")
chi2_arr, gamma_arr = read_txt("Images/param_gamma_chi_val.txt")
"""

def get_corner_plt(size, name):
    chi2_arr, Omega_arr = read_txt(f"Images_5/param_Omega_chi_val_{size}_{name}.txt")
    chi2_arr, alpha_arr = read_txt(f"Images_5/param_alpha_chi_val_{size}_{name}.txt")
    chi2_arr, gamma_arr = read_txt(f"Images_5/param_gamma_chi_val_{size}_{name}.txt")
    #chi2_arr, n_0_arr = read_txt(f"Images_4/param_n_0_chi_val_{size}_{name}.txt")

    """
    plt.scatter(Omega_arr, alpha_arr)
    plt.xlabel("Omega")
    plt.ylabel("Alpha")
    plt.title(f"Scatter of Omega and alpha pts = {size}")
    plt.savefig(f"Images_5/scatter_alpha_omega_{size}_{name}.png")
    plt.close()
    """
    #3
    #"""
    data = np.vstack([Omega_arr[size//10:], alpha_arr[size//10:], gamma_arr[size//10:], chi2_arr[size//10:]]).T
    param_num = 3
    labels = [r"$\Omega$", r"$\alpha$", r"$\gamma$", r"$\chi^2$"]
    #"""
    #2
    """
    data = np.vstack([Omega_arr[size//10:], n_0_arr[size//10:], chi2_arr[size//10:]]).T
    param_num = 2
    labels = [r"$\Omega$", r"$n_0$", r"$\chi^2$"]
    #"""
    #1
    """
    data = np.vstack([Omega_arr[size//10:], chi2_arr[size//10:]]).T
    param_num = 1
    labels = [r"$\Omega$", r"$\chi^2$"]
    #"""
    figure = corner.corner(data, labels = labels)
    plt.title(f"Corner size: {size}")
    plt.savefig(f"Images_5/corner_{param_num}_parameters_{size}_{name}.png")
    plt.close()

    #Plotting Chi2 against iterations
    plt.plot(chi2_arr)
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.yscale("log")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_5/{name}_Chi2.png", bbox_inches="tight")
    plt.close()


get_corner_plt(5000, "covariance_3_params")
#get_corner_plt(5000, "covariance_1_params")

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
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_3/{file_filename}_Chi2.png")
    plt.close()
    return mean_val, mean_paramters

#10 parameters
#"""
truth_dict_10 = {'x_0': 0.011887800744346281, 
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
#3 parameters
#"""
truth_dict_3 = {
'Omega': 77.65795555409711, 'alpha': 1.337069670593028, 'gamma': 0.9420617939335804, "Chi2":6900}
#"""
#1 parameters
#"""
truth_dict_1 = {
'Omega': 77.65795555409711, "Chi2":6900}
#"""

truth_10 = list(truth_dict_10.values())
truth_3 = list(truth_dict_3.values())
truth_1 = list(truth_dict_1.values())

#filename_2P_2K = "HMC_cloud_2_parameters_2000_iterations_0.1_noise_10_integration_step_size"
#filename_2P_10K = "HMC_cloud_2_parameters_10000_iterations_0.1_noise_10_integration_step_size"

filename_10P_5K = "HMC_cloud_10_parameters_5000_iterations_0.1_noise_10_integration_step_size"
filename_10P_10K = "HMC_cloud_10_parameters_10000_iterations_0.1_noise_10_integration_step_size"

#mean_val, mean_paramters = get_corner_plt_readfile(f"Images_6/{filename}.txt", filename, truth_10)
#mean_val, mean_paramters = get_corner_plt_readfile(f"Images_6/{filename}.txt", filename, truth_3)
#mean_val, mean_paramters = get_corner_plt_readfile(f"Images_6/{filename}.txt", filename, truth_1)

#Corner plots 10 parameters for 5K points
#Corner plot using txt files
def get_corner_plt_readfile_10P_5K(filename, file_filename, truth_dict_1, truth_dict_3, truth_dict_10, burn_in):
    with open(filename, 'r') as file:
        labels_10 = file.readline().strip().split()
    data = np.loadtxt(filename, skiprows = 1)
    labels_10 = [r"$x_0$", r"$y_0$", r"$z_0$", r"$i$", r"$\Omega$", r"$n_0$", r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\mu$", r"$\chi^2$"]
    labels_3 = [r"$\Omega$", r"$\alpha$", r"$\gamma$", r"$\chi^2$"]
    labels_1 = [r"$\Omega$",  r"$\chi^2$"]
    iterations = len(data[:,-1])
    #x_0 y_0 z_0 i Omega n_0 alpha beta gamma mu chi2
    # 0   1   2  3   4    5    6     7    8   9    10
    data_3 = data[:, [4, 6, 8, 10]]
    data_1 = data[:, [4, 10]]
    #Plotting Chi2 against iterations unfiltered
    plt.plot(data[:,-1])
    plt.title(f"Chi2 unfiltered")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_Chi2_unfiltered.png", bbox_inches="tight")
    plt.close()

    #Burn-in removal 
    #threshold = np.mean(data[:,-1])
    #filter = data[:, -1] < threshold
    #Burn in 10%-50% (???) take middle ground of 30%
    mean_val = np.mean(data, axis=0)
    mean_paramters = np.mean(data[:-1,:-1], axis=0)

    #10 parameters
    data_burn = data[burn_in:]
    result = np.vstack(data_burn)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_10, labels = labels_10, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_10_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"5K 10P complete")
    #3 parameters
    data_3 = data_3[burn_in:]
    result = np.vstack(data_3)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_3, labels = labels_3, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_3_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"5K 3P complete")
    #1 parameter
    data_1 = data_1[burn_in:]
    result = np.vstack(data_1)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_1, labels = labels_1, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_1_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"5K 1P complete")
    #Plotting Chi2 against iterations
    plt.plot(data[:,-1])
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.xlim(burn_in, len(data[:,-1]))
    #plt.ylim(data[:,-1][-1], data[:,-1][burn_in])
    plt.ylim(np.min(data[:,-1][burn_in:]), np.max(data[:,-1][burn_in:]))
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_Chi2.png", bbox_inches="tight")
    plt.close()
    return mean_val, mean_paramters


#Corner plots 10 parameters for 10K points

#Corner plot using txt files
def get_corner_plt_readfile_10P_10K(filename, file_filename, truth_dict_1, truth_dict_3, truth_dict_10, burn_in):
    with open(filename, 'r') as file:
        labels_10 = file.readline().strip().split()
    labels_10 = [r"$x_0$", r"$y_0$", r"$z_0$", r"$i$", r"$\Omega$", r"$n_0$", r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\mu$", r"$\chi^2$"]
    labels_3 = [r"$\Omega$", r"$\alpha$", r"$\gamma$", r"$\chi^2$"]
    labels_1 = [r"$\Omega$",  r"$\chi^2$"]
    data = np.loadtxt(filename, skiprows = 1)
    #x_0 y_0 z_0 i Omega n_0 alpha beta gamma mu chi2
    # 0   1   2  3   4    5    6     7    8   9    10
    #print(f"data shape: {np.shape(data)}")
    data_3 = data[:, [4, 6, 8, 10]]
    #print(f"data shape: {np.shape(data_3)}")
    data_1 = data[:, [4, 10]]
    iterations = len(data[:,-1])

    #Plotting Chi2 against iterations unfiltered
    plt.plot(data[:,-1])
    plt.title(f"Chi2 unfiltered")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_Chi2_unfiltered.png", bbox_inches="tight")
    plt.close()

    #Burn-in removal 
    #threshold = np.mean(data[:,-1])
    #filter = data[:, -1] < threshold
    #burn_in = 2500 #Burn in 10%-50% (???) take middle ground of 30%
    data_burn = data[burn_in:]
    mean_val = np.mean(data_burn, axis=0)
    #10
    mean_paramters = np.mean(data_burn[:-1,:-1], axis=0)
    result = np.vstack(data_burn)
    #print(f"result shape: {np.shape(result)}")
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_10, labels = labels_10, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_10_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"10K 10P complete")
    #3 parameters
    data_3 = data_3[burn_in:]
    result = np.vstack(data_3)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_3, labels = labels_3, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_3_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"10K 3P complete")
    #1 parameter
    data_1_burn = data_1[burn_in:]
    result = np.vstack(data_1_burn)
    #print(f"result = {result}")
    figure = corner.corner(result, truths=truth_dict_1, labels = labels_1, figsize=(30,30))
    #figure = corner.corner(result, labels = labels)
    plt.title(f"{iterations} data points")
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_1_parameters.png", bbox_inches="tight")
    plt.close()
    print(f"10K 1P complete")
    #Plotting Chi2 against iterations
    plt.plot(data[:,-1])
    plt.title(f"Chi2")
    plt.xlabel("Iterations")
    plt.ylabel("Chi2")
    #plt.yscale("log")
    plt.xlim(burn_in, len(data[:,-1]))
    #plt.ylim(data[:,-1][-1], data[:,-1][burn_in])
    plt.ylim(np.min(data[:,-1][burn_in:]), np.max(data[:,-1][burn_in:]))
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_Chi2.png", bbox_inches="tight")
    plt.close()

    #Plotting Omega against iterations
    plt.plot(data[:,4])
    plt.title(r"$\Omega$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\Omega$")
    #plt.yscale("log")
    plt.xlim(burn_in, len(data[:,4]))
    plt.ylim(np.min(data[:,4][burn_in:]), np.max(data[:,4][burn_in:]))
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_Omega.png", bbox_inches="tight")
    plt.close()

    #Plotting alpha against iterations
    plt.plot(data[:,6])
    plt.title(r"$\alpha$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\alpha$")
    #plt.yscale("log")
    plt.xlim(burn_in, len(data[:,6]))
    plt.ylim(np.min(data[:,6][burn_in:]), np.max(data[:,6][burn_in:]))
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_alpha.png", bbox_inches="tight")
    plt.close()

    #Plotting gamma against iterations
    plt.plot(data[:,8])
    plt.title(r"$\gamma$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$\gamma$")
    #plt.yscale("log")
    plt.xlim(burn_in, len(data[:,8]))
    plt.ylim(np.min(data[:,8][burn_in:]), np.max(data[:,8][burn_in:]))
    plt.tight_layout(pad=0.7)
    plt.savefig(f"Images_6/{file_filename}_gamma.png", bbox_inches="tight")
    plt.close()
    return mean_val, mean_paramters

burn_in = 8000 #Original
#burn_in = 1000
get_corner_plt_readfile_10P_10K(f"Images_6/{filename_10P_10K}.txt", filename_10P_10K, truth_1, truth_3, truth_10, burn_in)
burn_in = 2500
get_corner_plt_readfile_10P_5K(f"Images_6/{filename_10P_5K}.txt", filename_10P_5K, truth_1, truth_3, truth_10, burn_in)

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

