import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import sys
import os
#os.chdir("/Users/moskowitzi/Desktop/Senior_Year_at_Yale/git_repos/ASTR-375-Final-Project/")
#sys.path.append("/Users/moskowitzi/Desktop/Senior_Year_at_Yale/git_repos/ASTR-375-Final-Project/MoonPy")
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
import click


def mean_anomaly(P, e, omega_deg, b=0, a_over_R=None):
    omega = np.radians(omega_deg)

    if b == 0 or a_over_R is None:
        f = np.pi/2 - omega
    else:
        cos_f = -e + (1 + e*np.cos(omega)) * b / a_over_R
        # clamp into [-1, 1]
        cos_f = np.clip(cos_f, -1.0, 1.0)
        f = np.arccos(cos_f)

    E = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-e)/(1+e)))
    if E < 0:
        E += 2*np.pi
    M = E - e*np.sin(E)
    return (P / (2*np.pi)) * M

def kepler_3rd_law(mass_planet, mass_moon, a):
    G = 6.67*10**(-11) # in SI units
    return np.sqrt(4*np.pi**2/((mass_planet + mass_moon) * G) * a**3)
def mass_radius_polytrope(R): # assuming a polytropic equation of state for the rocky moon (not entirely accurate, but will do for now)
    # citation: https://iopscience.iop.org/article/10.3847/0004-637X/819/2/127
    #R = M**0.27 
    M = R**1/0.27 # this is a proportion, so the radius is described in Jupiter radii
    return M

def terrestrial_radius(mass, solar_radius): # make sure masses and radii are normalized to a reference point!
    M_e = 5.9722*10**24
    R_e = 6.3781*10**6
    return (mass/M_e)**(0.28) * R_e/solar_radius
def gas_giant_radius(mass):
    R_jup = 7.1492e7
    M_j = 1.898 * 10**27
    return (mass/M_j)**(-0.06)

def make_time_array(t0, P, N_epochs, B, cad_per_day):
    dt = 1.0 / cad_per_day
    all_times = []
    for n in range(N_epochs):
        center = t0 + n * P
        window = np.arange(center - B, center + B, dt)
        all_times.append(window)
    return np.concatenate(all_times)
    
    # make a list of (planet_mass, moon_mass, a_meters)

def transit_duration(P, aR, k, b, e=0.0, omega_deg=90.0):
    omega = np.radians(omega_deg)
    arg = ((1 + k)**2 - b**2) / (aR**2 - b**2)
    arg = np.clip(arg, 0,1)
    factor = np.sqrt(1 - e*e)/(1 + e*np.sin(omega))
    return (P/np.pi) * np.arcsin(np.sqrt(arg)) * factor


def simulate_plm(args):
    planet_mass, moon_mass, a_m, out_dir = args
    params = model_params()
    params.M_star = np.random.uniform(0.079, 2.0) * 1.989e30    # 0.08–2 M☉
    params.R_star = (params.M_star / 1.0e30)**0.8 * 6.957e8     # R ∝ M^0.8 [m]
    params.u1, params.u2 = 0.4089, 0.2556                      # limb darkening

    params.M_planet = planet_mass
    params.per_bary = kepler_3rd_law(params.M_star, planet_mass, a_m)  # days
    params.a_bary = a_m / params.R_star                          # in R_star
    if "terrestrial" in out_dir:
        params.r_planet = (planet_mass / 1.0e24)**0.28
    else:
        params.r_planet = gas_giant_radius(planet_mass)
    params.ecc_bary = np.random.uniform(0.0, 0.5)      
    params.b_bary   = np.random.uniform(0.0, 1.0 + params.r_planet)
    params.w_bary   = np.random.uniform(0, 360)                  
    params.t0_bary  = np.random.uniform(0, params.per_bary)    
    params.t0_bary_offset = 0

    hill_r = params.a_bary * (planet_mass / (3*params.M_star))**(1/3)
    moon_a = np.random.uniform(0.05*hill_r, 0.5*hill_r) * params.R_star
    params.M_moon   = moon_mass
    params.per_moon = kepler_3rd_law(planet_mass, moon_mass, moon_a) # days
    params.a_moon   = moon_a / params.R_star                     # in R_star
    params.r_moon   = (moon_mass / 1.0e24)**0.28                 # Rough scaling for terrestrial objects
    params.ecc_moon = np.random.uniform(0.0, 0.3)
    params.i_moon   = np.random.uniform(80, 90)
    params.Omega_moon = np.random.uniform(0, 360)
    params.w_moon     = np.random.uniform(0, 360)
    params.tau_moon   = np.random.uniform(0, params.per_moon)

    params.epochs             = 1
    params.epoch_distance     = params.per_bary
    params.epoch_duration     = 2.5 * (params.r_planet + 0.1) # extended the epoch range to cover the moon's transit as well
    params.cadences_per_day   = 250
    params.supersampling_factor = 1
    params.occult_small_threshold = 0.0
    params.hill_sphere_threshold  = 1.2

    t_grid = time(params).grid()                         # days
    model = moon_model(params)
    flux_total, flux_planet, flux_moon = model.light_curve(t_grid)

    noise_level = 50e-6                                              # 50 ppm
    flux_obs = flux_total + np.random.normal(0, noise_level, size=t_grid.shape)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
        f"planet{planet_mass}_moon{moon_mass}_a{a_m}.csv"
    )
    yerr = np.full(len(flux_obs), noise_level)
    data = {
        'time':t_grid,
        'flux_planet': flux_planet,
        'flux_moon': flux_moon,
        'flux_total': flux_total,
        't0_planet': params.t0_bary,
        't0_moon': params.tau_moon,
        'lc_data': flux_obs,
        'yerr': yerr
    }
    df = pd.DataFrame(data)
    #out_dir = "terrestrial_lightcurves/"
    df.to_csv(out_path)



@click.command()
@click.option(
    "--terrestrial/--no-terrestrial",
    default=False,
    help="Simulate terrestrial planet-moon systems"
)
@click.option(
    "--neptunian/--no-neptunian",
    default=False,
    help="Simulate neptunian planet-moon systems"
)
@click.option(
    "--jovian/--no-jovian",
    default=False,
    help="Simulate jovian planet-moon systems"
)
@click.option("--only_planet/--no-only_planet", default = False, help="Simulate only planetary systems (no moons)")
def main(terrestrial, neptunian, jovian, only_planet):
    solar_mass = 1.989 * 10**30 # kg
    solar_radius = 6.95700*10**8 # m
    
    min_star_mass = 0.079 * solar_mass
    min_star_radius = 0.102 * solar_radius
    
    max_star_mass = 6 * solar_mass
    max_star_radius = max_star_mass**0.8
    
    jupiter_mass = 1.898 * 10**27 # kg
    jupiter_radius = 7.1492*10**7 # m
    neptune_mass = 1.024 * 10**26 # kg
    neptune_radius = 2.4764 *10**7 # m
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
    min_moon_radius = 1.5608 * 10**6 # m Europa
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

    
    terrestrial_planet_list = terrestrial_mass_ranges
    neptunian_planet_list = neptunian_mass_ranges
    jovian_planet_list = jovian_mass_ranges
    moon_list   = moon_mass_ranges
    a_choices   = planet_a_ranges

    if terrestrial:
        planet_list = terrestrial_mass_ranges
        out_dir = "datasets/terrestrial_lightcurves"
        if only_planet:
            moon_list = np.zeros_like(planet_list)
            out_dir = f"{out_dir}_no_moon"
    elif neptunian:
        planet_list = neptunian_mass_ranges
        out_dir = "datasets/neptunian_lightcurves"
        if only_planet:
            moon_list = np.zeros_like(planet_list)
            out_dir = f"{out_dir}_no_moon"
    else:
        planet_list = jovian_mass_ranges
        out_dir = "datasets/jovian_lightcurves"
        if only_planet:
            moon_list = np.zeros_like(planet_list)
            out_dir = f"{out_dir}_no_moon"

    os.makedirs(out_dir, exist_ok=True)
    
    # make a list of (planet_mass, moon_mass, a_meters)
    tasks = [(pm, mm, np.random.choice(a_choices), out_dir) for pm, mm in product(planet_list, moon_list)]

    #for t in tasks[:5]:
        #simulate_plm(t)
    #print("✅ 5 example light‐curves generated successfully")
    
    n_workers = os.cpu_count() or 4
    
    #with ProcessPoolExecutor(max_workers=n_workers) as exe:
        #futures = [exe.submit(simulate_plm, t) for t in tasks]
        #for fut in tqdm(as_completed(futures), total=len(futures)):
            #_ = fut.result()    # raises if there was an exception
    results = process_map(simulate_plm,tasks,max_workers=os.cpu_count(),chunksize=100,desc="Simulating LC",unit="lc")
    
    print("Done!")

if __name__ == "__main__":
    main()
    #results = process_map(simulate_plm,tasks,max_workers=os.cpu_count(),desc="Simulating LC",unit="lc")
    
    print("Done!")



    

