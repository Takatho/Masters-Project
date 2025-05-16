# Example of using Zodipy

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

# Initialize a zodiacal light model at a wavelength/frequency or over a bandpass
model = zodipy.Model(25*u.micron)

"""
#Object has one required positional argument, 
#x, which can either represent a center wavelength/frequency or the points of an empirical bandpass. 
#If x represents the points of an instrument bandpass, the weights argument must also be provided
points = [3, 4, 5, 6] * u.micron
weights = [0.2, 0.4, 0.3, 0.1]

model = zodipy.Model(points, weights=weights)

#Selecting another model
model = zodipy.Model(25 * u.micron, name="planck13")
"""
# Use Astropy's `SkyCoord` object to specify coordinates
lon = [10, 10.1, 10.2] * u.deg
lat = [90, 89, 88] * u.deg
obstimes = Time(["2022-01-01 12:00:00", "2022-01-01 12:01:00", "2022-01-01 12:02:00"])
skycoord = SkyCoord(lon, lat, obstime=obstimes, frame="galactic")

"""
#3 frames: "barycentricmeanecliptic" (Ecliptic), "galactic" (Galactic), "icrs" (Celestial)
skycoord = SkyCoord(
    [40, 41, 42] * u.deg, 
    [60, 59, 58] * u.deg, 
    obstime=Time(["2020-01-01", "2020-01-02", "2020-01-03"]), 
    frame="galactic",)

"""
# Evaluate the zodiacal light model
emission = model.evaluate(skycoord)

"""
emission = model.evaluate(skycoord, obspos="mars")
print(emission)
# <Quantity [8.36985535] MJy / sr>

emission = model.evaluate(skycoord, obspos=[0.87, -0.53, 0.001] * u.AU)
print(emission)
# <Quantity [20.37750965] MJy / sr>
"""
print(emission)
#> [27.52410841 27.66572294 27.81251906] MJy / sr

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time, TimeDelta

from zodipy import Model

model = Model(30 * u.micron)

lats = np.linspace(-90, 90, 100) * u.deg
lons = np.zeros_like(lats)

t0 = Time("2022-06-14")
dt = TimeDelta(1, format="sec")
obstimes = t0 + np.arange(lats.size) * dt

coords = SkyCoord(
    lons,
    lats,
    frame=BarycentricMeanEcliptic,
    obstime=obstimes,
)

emission = model.evaluate(coords)

plt.plot(lats, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.savefig("Images/ecliptic_scan.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()

#"""
import multiprocessing

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

model = zodipy.Model(30 * u.micron)

nside = 256
pixels = np.arange(hp.nside2npix(nside))
lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

skycoord = SkyCoord(lon, lat, unit=u.deg, frame="galactic", obstime=Time("2022-01-14"))

emission = model.evaluate(skycoord, nprocesses=multiprocessing.cpu_count())

hp.mollview(
    emission,
    unit="MJy/sr",
    cmap="afmhot",
    min=0,
    max=80,
    title="Zodiacal light at 30 µm (2022-01-14)",
)
plt.savefig("Images/healpix_map.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()
#"""


import multiprocessing

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

COMP_NAMES = [
    "Smooth cloud",
    "Dust band 1",
    "Dust band 2",
    "Dust band 3",
    "Circum-solar Ring",
    "Earth-trailing Feature",
]

model = zodipy.Model(24 * u.micron)

nside = 128
pixels = np.arange(hp.nside2npix(nside))
lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

skycoord = SkyCoord(
    lon,
    lat,
    unit=u.deg,
    frame="barycentricmeanecliptic",
    obstime=Time("2022-01-14"),
)

emission = model.evaluate(skycoord, return_comps=True, nprocesses=multiprocessing.cpu_count())

fig = plt.figure(figsize=(8, 7))
for idx, comp_emission in enumerate(emission):
    hp.mollview(
        comp_emission,
        title=COMP_NAMES[idx],
        norm="log" if idx == 0 else None,
        cmap="afmhot",
        cbar=False,
        sub=(3, 2, idx + 1),
        fig=fig,
    )
plt.savefig("Images/component_maps.png", dpi=250, bbox_inches="tight")
#plt.show()
plt.close()



import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import LogNorm

from zodipy import grid_number_density

N = 200

x = np.linspace(-5, 5, N) * u.AU  # x-plane
y = np.linspace(-5, 5, N) * u.AU  # y-plane
z = np.linspace(-2, 2, N) * u.AU  # z-plane

density_grid = grid_number_density(
    x, y , z , obstime=Time("2021-01-01T00:00:00", scale="utc"),
    name="DIRBE",
)
density_grid = density_grid.sum(axis=0)  # Sum over all components

plt.pcolormesh(
    x.value,
    y.value,
    density_grid[N // 2].T,  # cross section in the yz-plane
    cmap="afmhot",
    norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
    shading="gouraud",
    rasterized=True,
)
plt.title("Cross section of the number density in the DIRBE model")
plt.xlabel("x [AU]")
plt.ylabel("z [AU]")
plt.savefig("Images/number_density.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()