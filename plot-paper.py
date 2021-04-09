import matplotlib.pylab as plt
import numpy as np
import matplotlib.gridspec as gridspec

import sys

import numpy as np
import matplotlib.pyplot as plt

from skimage import draw
from astropy.io import fits
from typing import Tuple, List, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_psf_stats(fn, extent_deg=0.25):
    if fn.endswith('npy'):
        im = np.load(fn)
        pixel_scale = 0.5/3600.
        num_pix = im.shape[0]
    else:
        with fits.open(fn) as hdulist:
            im = hdulist[0].data[0, 0]
            header = hdulist[0].header
            pixel_scale = abs(header['CDELT1'])
            num_pix = abs(header['NAXIS1'])
    max_radius = min(int(extent_deg/pixel_scale), num_pix//2)
    radial_average=np.zeros(max_radius)
    radial_std = np.zeros(max_radius)
    radial_maxabs = np.zeros(max_radius)
    for i in range(max_radius):
        circle_perimeter = draw.circle_perimeter(num_pix // 2, num_pix // 2, i)
        radial_average[i] = np.mean(abs(im[circle_perimeter]))
        radial_std[i] = np.std(im[circle_perimeter])
        radial_maxabs[i] = np.abs(im[circle_perimeter]).max()

    dist = np.linspace(0, max_radius*pixel_scale*3600, num=max_radius)
    return dist, radial_average, radial_std, radial_maxabs, im

def get_psf_stats_txt(fn, extent_deg, pixel_scale=0.5):
    im = np.load(fn)
#    im = im.reshape(int(len(im)**0.5), -1)
    num_pix = im.shape[0]
    max_radius = int(extent_deg/(pixel_scale*3600))
    max_radius = num_pix//2
    radial_average=np.zeros(max_radius)
    radial_std = np.zeros(max_radius)
    radial_maxabs = np.zeros(max_radius)

    for i in range(max_radius):
        circle_perimeter = draw.circle_perimeter(num_pix // 2, num_pix // 2, i)
        radial_average[i] = np.mean(abs(im[circle_perimeter]))
        radial_std[i] = np.std(im[circle_perimeter])
        radial_maxabs[i] = np.abs(im[circle_perimeter]).max()
    dist = np.linspace(0, max_radius*pixel_scale*60, num=max_radius)
    return dist, radial_average, radial_std, radial_maxabs    

# if __name__=='__main__':
#     fn = sys.argv[1]
#     extent_deg = np.float(sys.argv[2])
#     if fn[-4:]=="fits":
#         d, av, rstd, rmax = get_psf_stats(fn, extent_deg)#, pixel_scale=0.5)
#     else:
#         d, av, rstd, rmax = get_psf_stats_txt(fn, extent_deg)#, pixel_scale=0.5)
    
#     fnout=fn[:-5]+'out.npy'
#     np.save(fnout,np.concatenate([d, av, rstd, rmax]).reshape(4, -1))
#     exit()
# #    d = np.linspace(0, av.shape[0]*3600, av.shape[0])
#     plt.plot(d, np.abs(av))
#     plt.plot(d, rstd)
#     plt.plot(d, rmax)
#     plt.xlabel('arcminutes')
#     plt.semilogy()
#     plt.legend(['1chan','radial RMS','radial maximum'])
#     plt.show()

plt.rcParams.update({
                    'font.size': 12,
                    'font.family': 'serif',
                    'axes.labelsize': 14,
                    'axes.titlesize': 15,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'xtick.direction': 'in',
                    'ytick.direction': 'in',
                    'xtick.top': True,
                    'ytick.right': True,
                    'lines.linewidth': 0.5,
                    'lines.markersize': 5,
                    'legend.fontsize': 14,
                    'legend.borderaxespad': 0,
                    'legend.frameon': False,
                    'legend.loc': 'lower right'})

colors1 = ['k', '#482677FF', '#238A8DDF', '#95D840FF']

def plot_simulated_sky():
    galsize = np.random.gamma(2.25,1.5,10000) * 0.5 
    ellipticity=np.random.gamma(2,3, 10000)/20.*0.5
    ellipticity=np.random.gamma(10,10,10000)/1000.
    spec_ind = np.random.normal(-0.55, 0.15, 10000)

    nfluxhigh = np.random.uniform(0,0.1,100000)**(-2./3.)
    nfluxlow = np.random.uniform(0.05,1,100000)**(-1.)
    flux = nfluxhigh*nfluxlow

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(ellipticity, bins=50, color=colors1[1], alpha=0.5, density=True)
    ax1.set_xlabel('Ellipticity')
    ax1.set_yticklabels([])
    ax1.set_ylabel('Number')

    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(galsize, bins=50, color=colors1[2], alpha=0.5, density=True)
    ax2.set_xlabel('Semi-major axis [arcseconds]')
    ax2.set_yticklabels([])

    ax4 = plt.subplot(gs[1, 0])
    ax4.hist(np.log10(flux[flux>10.]), bins=100, log=True, color='C1', alpha=0.5, density=True)
    ax4.set_xlabel(r'$\log_{10}$(Flux [$\mu$ Jy])')
    ax4.set_yticklabels([])
    ax4.set_ylabel('Number')

    ax3 = plt.subplot(gs[1, 1])
    ax3.hist(spec_ind, color=colors1[3], bins=50, alpha=0.5, density=True)
    ax3.set_xlabel('Spectral index')
    ax3.set_yticklabels([])
    plt.tight_layout()

def plot_array(fncfg, fnpsf1, fnpsf2):
#    psf1 = np.load(fnpsf1)
#    psf2 = np.load(fnpsf2)
    d1,r1,rstd1,rmax1,psf1=get_psf_stats(fnpsf1, 0.5)
    d2,r2,rstd2,rmax2,psf2=get_psf_stats(fnpsf2, 0.5)
    
    psfplot = psf2
    nx = psfplot.shape[0]
    psfplot = psfplot[nx//2-1024:nx//2+1024, nx//2-1024:nx//2+1024]

    d = np.genfromtxt(fncfg)
    x, y = d[:,0], d[:, 1]
    nant = x.size
    print(d.shape)
    triuind = np.triu_indices(nant)

    # Compute baseline lengths 
    dX = x[:, None] - x[None]
    dY = y[:, None] - y[None]

    # Use subset of positions
    dx,dy = dX[triuind][:], dY[triuind][:]
    dy /= (3e2/1350.*1000.) # change units to kilo lambdas
    dx /= (3e2/1350.*1000.) # change units to kilo lambdas
    uv_density = np.histogram2d(dx,dy,bins=500)

    gs = gridspec.GridSpec(2, 6)
    ax1 = plt.subplot(gs[0, 0:2])
    ax1.scatter(x,y,c='k',s=1.5)
    ax1.set_title('DSA-2000\nantenna layout', weight='bold')
    ax1.set_xlabel('x position (m)')
    ax1.set_ylabel('y position (m)')

    ax2 = plt.subplot(gs[0, 2:4])
    uvext=[uv_density[1].min(), uv_density[1].max(),uv_density[2].min(),uv_density[2].max()]
    immy = ax2.imshow(np.log10(uv_density[0][::-1]), cmap='RdBu', extent=uvext, aspect='auto',)
#   ax2.text(30, 20, 'UV log(density)', )
    ax2.set_title('baseline log(density)\nsingle-channel snapshot',weight='bold')
    ax2.set_xlabel(r'$u$ (k$\lambda$)')
    ax2.set_ylabel(r'$v$ (k$\lambda$)')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(immy,cax=cax)

    ax3 = plt.subplot(gs[0, 4:])
    immy=ax3.imshow(np.log10(abs(psfplot)), cmap='Greys', extent=uvext, aspect='auto', vmax=0, vmin=-5.5)
    ax3.set_xlabel(r'$m$ (arcseconds)')
    ax3.set_ylabel(r'$l$ (arcseconds)')
    ax3.set_title('log(|PSF|)\nfull-band snapshot', weight='bold')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(immy,cax=cax)

    ax4 = plt.subplot(gs[1, 0:3])
    ax4.plot(d1,np.abs(r1),c='k')
    ax4.set_xlabel('Radius (arcseconds)')
    ax4.set_ylabel('abs(PSF) azimuthal average')
    ax4.legend(['single-channel snapshot'], loc=1)
    ax4.set_ylim(1e-7, 2.0)
    ax4.loglog()

    ax5 = plt.subplot(gs[1, 3:])
    ax5.plot(d2,np.abs(r2),c='darkblue')
    ax5.set_xlabel('Radius (arcseconds)')
    ax5.legend(['full band snapshot'], loc=1)
    ax5.set_ylim(1e-7, 2.0)
    ax5.loglog()

    tight_layout()

def plotdeconv(cmap='Greys', vmaxlr=10000, vminlr=1000, vmaxsr=5000, vminsr=0):
    nsub=3
    ax1 = plt.subplot(1,nsub,1)
    plt.title('Dirty map', color='C1', fontweight='bold', fontsize=15)
    #plt.axis('off')
    plt.imshow(datalr[...,0], cmap=cmap, vmax=vmaxlr, vmin=vminlr, 
               aspect='auto', extent=[0, datalr.shape[0]*2.0, datalr.shape[0]*2.0, 0])
    plt.setp(ax1.spines.values(), color='C1')
    plt.ylabel("m (arcseconds)")
    plt.xlabel("l (arcseconds)")

    ax2 = plt.subplot(1,nsub,2, sharex=ax1, sharey=ax1)
    plt.title('NN reconstruction', color='C2', 
              fontweight='bold', fontsize=15)
    plt.imshow(datasr[...,0], cmap=cmap, vmax=vmaxsr, vmin=vminsr, 
              aspect='auto', extent=[0, datasr.shape[0]*0.5,  datasr.shape[0]*0.5, 0])
    plt.setp(ax2.spines.values(), color='C2')
    plt.xlabel("l (arcseconds)")
#    plt.axis('off')
    # if calcpsnr:
    #   print("PSNR")
    #   plt.text(0.6, 0.85, psnr, color='white', fontsize=7, fontweight='bold')

    if nsub==3:
        ax5 = plt.subplot(1,nsub,3,sharex=ax1, sharey=ax1)
        plt.title('True map', color='k', fontweight='bold', fontsize=15)
        plt.imshow(datahr, cmap=cmap, vmax=vmaxsr, vmin=vminsr, aspect='auto', extent=[0, datasr.shape[0]*0.5, datasr.shape[0]*0.5, 0])
        plt.setp(ax5.spines.values(), color='k')
        plt.xlabel("l (arcseconds)")
       # plt.axis('off')


def plot_radial_psf(fnlr, fnhr, fnpsf, fn_model):
    from skimage import transform
    from model import resolve_single
    from model.edsr import edsr
    from utils import load_image, plot_sample
    from model.wdsr import wdsr_b
    from model.common import resolve_single16, tf
    import hr2lr

    psf = np.load(fnpsf)
    npsf = len(psf)

    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights(fn_model)

    fig = plt.figure(figsize=(15.5,12))

    gs = gridspec.GridSpec(3, 3)
#    gs.update(wspace=0.025, hspace=0.025)
    datahr = load_image(fnhr)
    datalr = load_image(fnlr)

    ymin,ymax = 656, 1088
    xmin,xmax = 444, 900

    rect = patches.Rectangle((xmin/4., ymin/4.), abs(0.25*(656-1088)), abs(0.25*(444-900)), linewidth=1.5, edgecolor='k', facecolor='none')

    ax1 = plt.subplot(gs[0:2, 0:2])
    ax1.imshow(datalr[:, :], vmax=7500, vmin=2500, cmap="Greys")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.add_patch(rect)
    ax1.set_xlim(50,450)
    ax1.set_ylim(50,450)

    ax2 = plt.subplot(gs[0, 2])
    ax2.imshow(datahr[ymin:ymax, xmin:xmax][::-1], vmax=1000, cmap='viridis')
    ax2.set_xticks([])
    ax2.set_yticks([])

    datahr = load_image(fnhr)
    datasr = resolve_single16(model, datalr)
    ax3 = plt.subplot(gs[1, 2])
    ax3.imshow(datasr[ymin:ymax, xmin:xmax, 0][::-1], vmax=1000, cmap="viridis")
    ax3.set_xticks([])
    ax3.set_yticks([])

    psf_stretch = transform.rescale(psf, 1.05)
    psf_stretch = psf_stretch[len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2,
                             len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2]

    datahr = load_image(fnhr)
    datalr = hr2lr.convolvehr(datahr, psf_stretch, noise=False, nbit=16)
    datasr = resolve_single16(model, datalr)
    ax4 = plt.subplot(gs[2, 2])
    ax4.imshow(datasr[ymin:ymax, xmin:xmax,0][::-1], vmax=1000, cmap="viridis")
    ax4.set_xticks([])
    ax4.set_yticks([])

    psf_stretch = transform.rescale(psf, 1.25)
    psf_stretch = psf_stretch[len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2,
                             len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2]

    datahr = load_image(fnhr)
    datalr = hr2lr.convolvehr(datahr, psf_stretch, noise=False, nbit=16)
    datasr = resolve_single16(model, datalr)
    ax5 = plt.subplot(gs[2, 1])
    ax5.imshow(datasr[ymin:ymax, xmin:xmax,0][::-1], vmax=1000, cmap="viridis")
    ax5.set_xticks([])
    ax5.set_yticks([])

    psf_stretch = transform.rescale(psf, 2.0)
    psf_stretch = psf_stretch[len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2,
                             len(psf_stretch)//2-npsf//2:len(psf_stretch)//2+npsf//2]

    datahr = load_image(fnhr)
    datalr = hr2lr.convolvehr(datahr, psf_stretch, noise=False, nbit=16)
    datasr = resolve_single16(model, datalr)
    ax6 = plt.subplot(gs[2, 0])
    ax6.imshow(datasr[ymin:ymax, xmin:xmax,0][::-1], vmax=1000, cmap="viridis")

    ax6.set_xticks([])
    ax6.set_yticks([])

    plt.setp(ax1.spines.values(), color='C1', lw=4)
    plt.setp(ax2.spines.values(), color='k',  lw=4)
    plt.setp(ax3.spines.values(), color='C2', lw=4)
    plt.setp(ax4.spines.values(), color='C2', lw=4)
    plt.setp(ax5.spines.values(), color='C2', lw=4)
    plt.setp(ax6.spines.values(), color='C2', lw=4)


    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # place a text box in upper left in axes coords
    ax1.text(0.85, 0.05, r'Dirty image', 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    ax2.text(0.7, 0.1, r'True sky', 
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    # place a text box in upper left in axes coords
    ax3.text(0.7, 0.1, r'PSF$_{true}$=PSF$_{train}$', 
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    # place a text box in upper left in axes coords
    ax4.text(0.7, 0.1, r'PSF$_{true}$=1.05$\times$PSF$_{train}$', 
            transform=ax4.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    # place a text box in upper left in axes coords
    ax5.text(0.7, 0.1, r'PSF$_{true}$=1.25$\times$PSF$_{train}$', 
            transform=ax5.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)
    # place a text box in upper left in axes coords
    ax6.text(0.7, 0.1, r'PSF$_{true}$=2$\times$PSF$_{train}$', 
            transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center', bbox=props)













