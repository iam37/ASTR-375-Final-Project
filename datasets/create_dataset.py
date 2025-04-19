import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import sys
import os
os.chdir("/Users/moskowitzi/Desktop/Senior_Year_at_Yale/git_repos/ASTR-375-Final-Project/")
sys.path.append("/Users/moskowitzi/Desktop/Senior_Year_at_Yale/git_repos/ASTR-375-Final-Project/MoonPy")
#from moonpy import *
#from moonpy import LightCurve
import batman
from batman import TransitParams
from batman import TransitModel

from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from tqdm import tqdm
import glob


from lightkurve import search_lightcurvefile
from lightkurve import LightCurveCollection
from lightkurve import LightCurve

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.contrib.concurrent import process_map

from pandoramoon import pandora, model_params, moon_model, time

#from moonpy import transitmodel as tm 


def mean_anomaly(P, e, omega_deg, b=0, a_over_R=None):
    omega = np.radians(omega_deg)

    if b == 0 or a_over_R is None:
        f = np.pi / 2 - omega
    else:
        cos_f = -e + (1 + e * np.cos(omega)) * b / a_over_R
        f = np.arccos(cos_f)
    #f_transit = np.pi / 2 - omega
    E = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - e) / (1 + e)))
    if E < 0:
        E += 2 * np.pi
    M = E - e * np.sin(E)
    delta_t = (P / (2 * np.pi)) * M
    return delta_t

def kepler_3rd_law(mass_planet, mass_moon, a):
    G = 6.67*10**(-11) # in SI units
    return np.sqrt(4*np.pi**2/((mass_planet + mass_moon) * G) * a**3)
def mass_radius_polytrope(R): # assuming a polytropic equation of state for the rocky moon (not entirely accurate, but will do for now)
    # citation: https://iopscience.iop.org/article/10.3847/0004-637X/819/2/127
    #R = M**0.27 
    M = R**1/0.27 # this is a proportion, so the radius is described in Jupiter radii
    return M

def terrestrial_radius(mass): # make sure masses and radii are normalized to a reference point!
    return mass**(0.28)

def make_time_array(t0, P, N_epochs, B, cad_per_day):
    """Return a (small!) array that covers N_epochs windows of
       length (2B + T_d), each centered on t0 + n⋅P."""
    dt = 1.0 / cad_per_day
    all_times = []
    for n in range(N_epochs):
        center = t0 + n * P
        window = np.arange(center - B, center + B, dt)
        all_times.append(window)
    return np.concatenate(all_times)

def simulate_plm(args):
    solar_mass = 1.989 * 10**30 # kg
    solar_radius = 6.95700*10**8 # m
    planet_mass, moon_mass, a_meters = args
    #for a_meters in tqdm(planet_a_ranges):
    #a_meters = np.random.choice(planet_a_ranges, size=1)[0]
    #print(a_meters)
    params = model_params()
    params.R_star = solar_radius  # [m]
    params.u1 = 0.4089
    params.u2 = 0.2556
    a_pl = a_meters / params.R_star
    eccentricity = np.random.rand()
    params.ecc_bary = eccentricity 
    params.a_bary = a_pl
    params.b_bary = np.random.rand()
    planet_radius = terrestrial_radius(planet_mass/solar_mass) # in stellar units
    params.r_planet = planet_radius
    period = kepler_3rd_law(solar_mass, planet_mass, a_meters)
    params.per_bary = period * 1/3600 * 1/24
    params.w_bary = 20
    params.t0_bary = mean_anomaly(period, eccentricity, params.w_bary, b=params.b_bary, a_over_R=a_pl)
    params.t0_bary_offset = 0
    params.M_planet = planet_mass


    # Moon parameters
    moon_radius = terrestrial_radius(moon_mass/solar_mass) # assuming rocky moons that have a similar empirical relation as rocky planets
    hill_radius = params.a_bary * (params.M_planet/(3*(solar_mass+params.M_planet)))**(1/3)
    params.r_moon = moon_radius
    moon_a = np.random.uniform(0.01*hill_radius, 0.5*hill_radius)
    params.per_moon = kepler_3rd_law(planet_mass, moon_mass, moon_a) * 1/3600 * 1/24
    params.Omega_moon = 0
    params.tau_moon = mean_anomaly(params.per_moon, 0.001, params.Omega_moon)
    params.w_moon = 20
    params.e_moon = 0.001
    params.i_moon = np.random.uniform(80, 90)
    params.M_moon = moon_mass

    #params.epoch_duration = 0.6  # [days]
    params.epochs = 1

    B = 0.5
    t_start = params.t0_bary - B
    t_end   = params.t0_bary + B
    
    k = params.r_planet
    arg = (1 + k)**2 - params.b_bary**2
    
    duration = (params.per_bary/np.pi
           * np.arcsin((1/params.a_bary) * np.sqrt(arg))
           * np.sqrt(1 - params.ecc_bary**2)
           / (1 + params.ecc_bary*np.sin(np.radians(params.w_bary))))

    params.epoch_distance = params.per_bary
    params.epoch_duration = duration
    params.cadences_per_day = 250
    cadence = 1/params.cadences_per_day  # days per sample
    time_array = np.arange(t_start, t_end, cadence)
    """params.epoch_duration = T_d + 2*B
    #params.epoch_duration = duration
    #print(transit_time)
    #params.epoch_duration = transit_time
    params.cadences_per_day = 250  # [int]
    params.epoch_distance = params.per_bary   # [days]
    params.supersampling_factor = 1  # [int]
    params.occult_small_threshold = 0.1  # [0..1]
    params.hill_sphere_threshold = 1.2"""


    #time_array = time(params).grid()
    
    model = moon_model(params)
    flux_total, flux_planet, flux_moon = model.light_curve(time_array)
    #print("flux_total", flux_total)
    #print("flux_planet", flux_planet)
    #print("flux_moon", flux_moon)

    xp, yp, xm, ym = model.coordinates(time_array)
    noise_level = 100e-6 # Gaussian noise -- would this reflect the noise-profile of TESS?
    noise = np.random.normal(0, noise_level, len(time_array))
    test_data = noise + flux_total
    #tensor_input = np.expand_dims(tensor_input, axis=-1)
    yerr = np.full(len(test_data), noise_level)
    data = {
        'time':time_array,
        'flux_planet': flux_planet,
        'flux_moom': flux_moon,
        'flux_total': flux_total,
        't0_planet': params.t0_bary,
        't0_moon': params.tau_moon,
        'lc_data': test_data,
        'yerr': yerr
    }
    df = pd.DataFrame(data)
    out_dir = "terrestrial_lightcurves/"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}a_{a_meters}_planet_mass_{planet_mass}_moon_mass_{moon_mass}.csv")

def main():
    solar_mass = 1.989 * 10**30 # kg
    solar_radius = 6.95700*10**8 # m
    
    min_star_mass = 0.079 * solar_mass
    min_star_radius = 0.102 * solar_radius
    
    max_star_mass = 6 * solar_mass
    max_star_radius = max_star_mass**0.8
    
    jupiter_mass = 1.898 * 10**27 # kg
    jupiter_radius = 7.1492*10**7 # m
    neptune_mass = 1.024 * 10**26 # kg
    neptune_radius = 2.4764 *10**3 # m
    earth_mass = 5.9722*10**24 #kg
    earth_radius = 6.3781*10**6 # m
    super_earth_mass = 10*earth_mass # guides for super_earth parameters, numbers vary
    super_earth_radius = 1.2 * earth_radius
    
    # Orbital characteristics
    min_a = 0.2877 * 1.496 * 10**11 # orbital radius in meters
    max_a = 778479000000 # orbit of Jupiter in meters
    
    # Moon characteristics
    min_moon_mass = 4.79984 * 10**22 # Europa, kg
    max_moon_mass = earth_mass
    min_moon_radius = 1.5608 * 10**3 # m Europa
    max_moon_radius = earth_radius
    
    moon_mass_ranges = 10**(np.random.uniform(np.log10(min_moon_mass), np.log10(max_moon_mass), size=200))
    #planet_mass_ranges =np.random.uniform(super_earth_mass, 10*jupiter_mass, size = 2000)
    log_earth = np.log10(earth_mass)
    log_super_earth = np.log10(super_earth_mass)
    terrestrial_mass_ranges = 10**(np.random.uniform(log_earth, log_super_earth, size=200))
    np.sort(terrestrial_mass_ranges)
    
    neptunian_mass_ranges = 10**(np.random.uniform(log_super_earth, np.log10(jupiter_mass), size = 200))
    np.sort(neptunian_mass_ranges)
    
    jovian_mass_ranges = 10**(np.random.uniform(np.log10(jupiter_mass), np.log10(10*jupiter_mass), size=200))
    np.sort(jovian_mass_ranges)
    
    planet_a_ranges = 10**(np.random.uniform(np.log10(min_a), np.log10(max_a), size = 200))
    solar_radii = 10**(np.random.uniform(np.log10(min_star_radius), np.log10(max_star_radius), size = 200))

    
    planet_list = terrestrial_mass_ranges
    moon_list   = moon_mass_ranges
    a_choices   = planet_a_ranges
    
    # make a list of (planet_mass, moon_mass, a_meters)
    tasks = [
        (pm, mm, np.random.choice(a_choices))
        for pm, mm in product(planet_list, moon_list)
    ]
    
    """n_workers = os.cpu_count() or 4
    
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(simulate_plm, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            _ = fut.result()    # raises if there was an exception"""
    results = process_map(simulate_plm,tasks,max_workers=os.cpu_count(),desc="Simulating LC",unit="lc")
    
    print("Done!")

if __name__ == "__main__":
    main()



    

