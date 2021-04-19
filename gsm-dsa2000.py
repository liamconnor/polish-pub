from pygdsm import GlobalSkyModel
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits
import os

def generate_gsm(freq=700.,output_file='gsm.fits',plot=False):
    """ 
    makes map from GSM
    freq (MHz): gsm freq
    output_file (essential): fits file to write gsm to
    plot (bool): display map 
    """

    os.system("rm -rf "+output_file)
    gsm = GlobalSkyModel()
    gsm.generate(freq)
    if plot:
        gsm.view(logged=True)
    gsm.write_fits(output_file)
    
def generate_sim(file='gsm.fits',plot=False,gal_cut=10.0,smoothing=1.0,inner_scale=0.):
    """ 
    returns scrambled map from input gsm file. units are K
    file (essential): gsm fits file
    plot (bool): display map 
    gal_cut (deg): ignore |b|<gal_cut for alms
    smoothing (deg): smooth output map on this scale
    inner_scale (deg): discard alm power on larger scales
    """
    
    map = hp.read_map(file)
    cl = hp.anafast(map,gal_cut=gal_cut)
    
    if inner_scale > 0.:
        l_cut = int(np.floor(180./inner_scale))+1
    cl[:l_cut] = 0.
    
    print(cl)
    map_syn = hp.synfast(cl,hp.get_nside(map))
    map_syn = hp.smoothing(map_syn,fwhm=smoothing*np.pi/180.)
    if plot:
        hp.mollview(map_syn)
        plt.title('Synthetic map')
        plt.show()
    return map_syn

def create_wcs(nside=512,side_deg=10.0):
    """ 
    returns wcs for cutout at random sky location
    nside: number of pixels per side
    side_deg: length of side in deg
    """
    
    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)

    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.crpix = [nside/2,nside/2]
    ra = 2.*np.pi*np.random.uniform()*180./np.pi
    dec = np.arcsin(np.random.uniform()*2.-1.)*180./np.pi
    print("RA: ",ra," -- DEC: ",dec)
    w.wcs.cdelt = np.array([-side_deg/nside, side_deg/nside])
    w.wcs.crval = [ra,dec]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # Now, write out the WCS object
    return w

def interp_sim(map_syn,w,nside=512):
    """ 
    returns interpolated map from synthetic sky. units are K
    w: wcs
    nside: number of pixels per side
    """
    
    x,y = np.meshgrid(np.arange(nside),np.arange(nside))
    c = w.pixel_to_world(x,y)
    m = hp.get_interp_val(map_syn,c.dec.value,c.ra.value,lonlat=True)
    return m

def plot_interp(m,w):
    """
    plots interpolated map
    m: interpolated map array
    w: wcs
    """
    
    plt.subplot(projection=w)
    plt.imshow(m,origin='lower')
    plt.grid(color='white', ls='solid')
    plt.title('Snapshot')
    plt.colorbar()
    plt.show()


if __name__=='__main__':
	# generate gsm and write to disk
	generate_gsm()
	# generate simulated map with spherical harmonics scrambled
	map_syn = generate_sim(plot=True,inner_scale=2.)
	# make wcs for cutout
	w = create_wcs()
	# make cutout
	m = interp_sim(map_syn,w)
	# plot
	plot_interp(m,w)

