import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage import draw
from astropy.io import fits
from typing import Tuple, List, Optional
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

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

def get_psf_stats(fn, extent_deg=0.25, pixel_scale=0.5):
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

    print(pixel_scale*3600)
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


def plot_simulated_sky():
    galsize = np.random.gamma(2.25,1.,10000) * 0.5 
    ellipticity=np.random.gamma(2,3, 10000)/20.*0.5
    ellipticity=np.random.gamma(10,10,10000)/1000.
    ellipticity=np.random.beta(1.7,4.5,10000)
    spec_ind = np.random.normal(-0.55, 0.15, 10000)

    nfluxhigh = np.random.uniform(0,0.1,100000)**(-2./3.)
    nfluxlow = np.random.uniform(0.05,1,100000)**(-1.)
    flux = nfluxhigh*nfluxlow

    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax1.hist(ellipticity, bins=50, color=colors1[1], alpha=0.25, density=True)
    ax1.set_xlabel('Ellipticity')
    ax1.set_yticklabels([])
    ax1.set_ylabel('Number')

    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(galsize, bins=50, color=colors1[2], alpha=0.25, density=True)
    ax2.set_xlabel('Semi-major axis [arcseconds]')
    ax2.set_yticklabels([])

    ax4 = plt.subplot(gs[1, 0])
    ax4.hist(np.log10(flux[flux>10.]), bins=100, log=True, color='C1', alpha=0.25, density=True)
    ax4.set_xlabel(r'$\log_{10}$(Flux [$\mu$ Jy])')
    ax4.set_yticklabels([])
    ax4.set_ylabel('Number')

    ax3 = plt.subplot(gs[1, 1])
    ax3.hist(spec_ind, color=colors1[3], bins=50, alpha=0.25, density=True)
    ax3.set_xlabel('Spectral index')
    ax3.set_yticklabels([])
    plt.tight_layout()


def plot_array(fncfg, fnpsf1, fnpsf2):
#    psf1 = np.load(fnpsf1)
#    psf2 = np.load(fnpsf2)
    d1,r1,rstd1,rmax1,psf1=get_psf_stats(fnpsf1, 0.5)
    d2,r2,rstd2,rmax2,psf2=get_psf_stats(fnpsf2, 0.5)
    
    pixel_scale = np.diff(d1)[0]

    psfplot = psf2
    nx = psfplot.shape[0]
    psfplot = psfplot[nx//2-1024:nx//2+1024, nx//2-1024:nx//2+1024]

    d = np.genfromtxt(fncfg)
    x, y = d[:,0], d[:, 1]
    nant = x.size
    print(d.shape)
    triuind = np.triu_indices(nant, 1)

    # Compute baseline lengths 
    dX = x[:, None] - x[None]
    dY = y[:, None] - y[None]

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
    immy=ax3.imshow(np.log10(abs(psfplot)), cmap='Greys', 
                    extent=[-0.5*pixel_scale*nx, 0.5*pixel_scale*nx, -0.5*pixel_scale*nx, 0.5*pixel_scale*nx], 
                    aspect='auto', vmax=0, vmin=-5.5)
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    ax3.set_xlabel(r'$m$ (arcseconds)')
    ax3.set_ylabel(r'$l$ (arcseconds)')
    ax3.set_title('log(|PSF|)\nfull-band 15 minutes', weight='bold')
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
    ax5.legend(['full band 15 minutes'], loc=1)
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


def plot_example_sr(datalr, datasr, datahr=None, dataother=None,
            cmap='Greys', suptitle=None, 
            fnfigout='test.pdf', vm=None, nbit=16, 
            calcpsnr=True, vmsr=None, vmc=None):

    fig=plt.figure(figsize=(11.75,8.8))

    datalr_1, datalr_2 = datalr 
    datahr_1, datahr_2 = datahr 
    datasr_1, datasr_2 = datasr

    if datahr is None:
        nsub=2
    else:
        nsub=3
    if datahr is not None:
      pass

    if dataother is not None:
        nsub += 1
        dataclean_1, dataclean_2 = dataother

    if calcpsnr:
        psnr_2 = tf.image.psnr(datasr_1[None, ...,None].astype(np.uint16), 
                             datahr_1[None, ..., None].astype(np.uint16), 
                             max_val=2**(nbit)-1)
        ssim_2 = tf.image.ssim(datasr_1[None, ..., None].astype(np.uint16), 
                             datahr_1[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_2 = "       1.5'' - \nPSNR = %0.1f\nSSIM = %0.4f" % (psnr_2, ssim_2)

        psnr_1 = tf.image.psnr(datasr_2[None, ...,None].astype(np.uint16), 
                             datahr_2[None, ..., None].astype(np.uint16), 
                            max_val=2**(nbit)-1)
        ssim_1 = tf.image.ssim(datasr_2[None, ..., None].astype(np.uint16), 
                             datahr_2[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_1 = "       1.5'' - \nPSNR = %0.1f\nSSIM = %0.4f" % (psnr_1, ssim_1)

        psnr_clean_1 = tf.image.psnr(dataclean_1[None, ...,None].astype(np.uint16), 
                             datahr_1[None, ..., None].astype(np.uint16), 
                            max_val=2**(nbit)-1)
        ssim_clean_1 = tf.image.ssim(dataclean_2[None, ..., None].astype(np.uint16), 
                             datahr_2[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_clean_1 = "       1.5'' - \nPSNR = %0.1f\nSSIM = %0.4f" % (psnr_clean_1, ssim_clean_1)

        psnr_clean_2 = tf.image.psnr(dataclean_1[None, ...,None].astype(np.uint16), 
                             datahr_1[None, ..., None].astype(np.uint16), 
                            max_val=2**(nbit)-1)
        ssim_clean_2 = tf.image.ssim(dataclean_1[None, ..., None].astype(np.uint16), 
                             datahr_1[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_clean_2 = "       1.5'' - \nPSNR = %0.1f\nSSIM = %0.4f" % (psnr_clean_2, ssim_clean_2)


    if vm is None:
      vminlr=max(0.9*np.median(datalr_1), 0)
      vmaxlr=np.median(datalr_1)+0.01*(np.max(datalr_1)-np.median(datalr_1))

      vminsr=max(0.9*np.median(datasr), 0)
      vmaxsr=np.median(datasr_1)+0.01*(np.max(datasr_1)-np.median(datasr_1))

      vminhr=max(0.9*np.median(datahr), 0)
      vmaxhr=np.median(datalr_1)+0.01*(np.max(datalr_1)-np.median(datalr_1))
    else:
      vminlr, vminsr, vminhr = 0, 0, 0
      vmaxlr, vmaxsr, vmaxhr = vm, vm, vm

    vmaxlr=5000
    vminlr_2=3000

    xlim1 = np.random.uniform(0.2,0.8)
    ylim1 = np.random.uniform(0.2,0.8)
    xlim1, ylim1 = 0.46029491318597404, 0.5403755860037598
    xlim1, ylim1 = 0.823, 0.695

    dx, dy = 0.12, 0.12, #0.16, 0.16

    xlim2 = 0.6436#np.random.uniform(0.2,0.8)
    ylim2 = 0.7150#np.random.uniform(0.2,0.8)
    dx2, dy2 = .14, .14

    ax1 = plt.subplot(2,nsub,nsub+1)
    ax1.set_yticks([])
    ax1.set_xticks([])
    plt.ylabel('1300 MHz PSF', labelpad=40,fontsize=20)    
    plt.imshow(datalr_1, cmap=cmap, vmax=0.5*vmaxlr, vmin=vminlr, 
               aspect='auto', extent=[0,1,0,1])
    plt.xlim(xlim1,xlim1+dx)
    plt.ylim(ylim1,ylim1+dy)
    plt.setp(ax1.spines.values(), color='C1')

    ax2 = plt.subplot(2,nsub,nsub+2, sharex=ax1, sharey=ax1)
    plt.imshow(datasr_1, cmap=cmap, vmax=0.9*vmaxsr, vmin=vminsr, 
              aspect='auto', extent=[0,1,0,1])
    plt.xlim(xlim1,xlim1+dx)
    plt.ylim(ylim1,ylim1+dy)
    ax1.set_yticks([])
    ax1.set_xticks([])

    if calcpsnr:
        print("PSNR")
#        plt.text(xlim2+0.015, ylim2+0.005, psnr_2, color='C3', fontsize=9, fontweight='bold')
        props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='white')
        # place a text box in upper left in axes coords
        plt.text(xlim1+0.01, ylim1+0.005, psnr_2, color='C3', fontsize=9, 
            fontweight='bold', bbox=props)

    ax5 = plt.subplot(2,nsub,nsub+3,sharex=ax1, sharey=ax1)
    plt.imshow(datahr_1, cmap=cmap, vmax=0.9*vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])
    plt.xlim(xlim1,xlim1+dx)
    plt.ylim(ylim1,ylim1+dy)
    ax1.set_yticks([])
    ax1.set_xticks([])

    ax3 = plt.subplot(2,nsub,1)
    ax3.imshow(datalr_2, cmap=cmap, vmax=vmaxlr, vmin=vminlr_2, 
              aspect='auto', extent=[0,1,0,1])
#    plt.title('Dirty map \nzoom', color='C1', fontweight='bold', fontsize=15)
    plt.title('Dirty image', color='C1', fontweight='bold', fontsize=15, pad=20)

    ax3.set_yticks([])
    ax3.set_xticks([])
    plt.ylabel('10 MHz PSF', labelpad=40, fontsize=20) 
    plt.xlim(xlim2,xlim2+dx2)
    plt.ylim(ylim2,ylim2+dy2)

    ax4 = plt.subplot(2,nsub,2,sharex=ax3, sharey=ax3)
#    plt.title('NN reconstruction\nzoom ', color='C2', 
#              fontweight='bold', fontsize=15)

    ax4.imshow(datasr_2, cmap=cmap, 
              vmax=vmaxsr, vmin=vminsr, aspect='auto', extent=[0,1,0,1])

    plt.suptitle(suptitle, color='C0', fontsize=20)

    ax4.set_yticks([])
    ax4.set_xticks([])
    plt.xlim(xlim2,xlim2+dx2)
    plt.ylim(ylim2,ylim2+dy2)

    plt.title('POLISH\nreconstruction', color='C2', 
              fontweight='bold', fontsize=15, pad=10)

    if calcpsnr:
        print("PSNR")
        props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='white')
        plt.text(xlim2+0.01, ylim2+0.005, psnr_1, color='C3', fontsize=9, fontweight='bold', bbox=props)

    ax6 = plt.subplot(2,nsub,3,sharex=ax3, sharey=ax3)
#    plt.title('True sky', color='k', fontweight='bold', fontsize=15)      
    print(datahr_2.shape, datahr_2.sum())  
    ax6.imshow(datahr_2[:,:], cmap=cmap, 
               vmax=vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])
    ax6.set_yticks([])
    ax6.set_xticks([])
    plt.title('True sky', color='k', fontweight='bold', fontsize=15, pad=20)
    plt.xlim(xlim2,xlim2+dx2)
    plt.ylim(ylim2,ylim2+dy2)

    ax7 = plt.subplot(2,nsub,4,sharex=ax3, sharey=ax3)
    ax7.imshow(dataclean_2, cmap=cmap, 
               vmax=vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])

    plt.xlim(xlim2,xlim2+dx2)
    plt.ylim(ylim2,ylim2+dy2)

    if calcpsnr:
        props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='white')
        plt.text(xlim2+0.01, ylim2+0.005, psnr_clean_2, color='C3', fontsize=9, fontweight='bold', bbox=props)

    plt.title('CLEAN', color='C3', fontweight='bold', fontsize=15, pad=20)

    ax8 = plt.subplot(2,nsub,nsub+4,sharex=ax1, sharey=ax1)
    ax8.imshow(dataclean_1, cmap=cmap, vmax=0.9*vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])
    plt.xlim(xlim1,xlim1+dx)
    plt.ylim(ylim1,ylim1+dy)

    if calcpsnr:
        props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='white')
        plt.text(xlim1+0.01, ylim1+0.005, psnr_clean_1, color='C3', fontsize=9, fontweight='bold', bbox=props)

    plt.setp(ax3.spines.values(), color='C1', lw=2)
    plt.setp(ax6.spines.values(), color='k',  lw=2)
    plt.setp(ax4.spines.values(), color='C2', lw=2)
    plt.setp(ax1.spines.values(), color='C1', lw=2)
    plt.setp(ax5.spines.values(), color='k',  lw=2)
    plt.setp(ax2.spines.values(), color='C2', lw=2)
    plt.setp(ax7.spines.values(), color='C3', lw=2)
    plt.setp(ax8.spines.values(), color='C3', lw=2)
    plt.tight_layout()
    plt.show()

def run_plot_example_sr():
    lf=np.load('./plots/lr-fullband-0818.npy')
    hf=np.load('./plots/hr-fullband-0818.npy')
    sf=np.load('./plots/sr-fullband-0818.npy')
    cf=hf

    l1=np.load('./plots/lr-1chan-0813.npy')
    h1=np.load('./plots/hr-1chan-0813.npy')
    s1=np.load('./plots/sr-1chan-0813.npy')
    c1=np.load('./plots/cl-1chan-0813.npy')

    plot_example_sr((lf,l1), (sf,s1), (hf,h1), dataother=(cf,c1), 
                     vm=1900, calcpsnr=True, cmap='Greys',)
#    plt.savefig('example_polish.pdf')


def perturbation_figure(hr, psf, model):

    rad_stretch = np.linspace(1., 2.0, 4)
    psnr_arr,ssim_arr=[],[]
    for rr in rad_stretch:
        lr = hr2lr.convolvehr(hr, transform.rescale(psf,rr), nbit=16, rebin=4)

        dsr = (resolve_single(model,lr)).numpy()

        psnr_ = tf.image.psnr(dsr[None, ...,0,None].astype(np.uint16),
                      hr[None, ..., None].astype(np.uint16),
                      max_val=2**(nbit)-1).numpy()[0]

        ssim = tf.image.ssim(dsr[None, ...,0, None].astype(np.uint16), 
                             hr[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        print(rr, psnr_)
        psnr_arr.append(psnr_)
        ssim_arr.append(ssim)

    return rad_stretch, ssim_arr, psnr_arr

def psf_perturbation_plot():
    rad_stretch = np.linspace(1., 1.35, 3)
    nuggets = np.linspace(0, 30.0, 3)
    psnr_arr,ssim_arr=[],[]
    kk=0
    gs = gridspec.GridSpec(3,9)
    gs.update(wspace=0.05, hspace=0.05)
    for ii,rr in enumerate(rad_stretch):
        for jj,nn in enumerate(nuggets):
            psf_ = transform.rescale(psf,rr)
            psf_ = elastic_transform(psf_[:,:,None]*np.ones([1,1,3]), alpha=nn,
                                     sigma=3, alpha_affine=0)[:,:,0]

            n = psf_.shape[0]//2
            plt.subplot(gs[ii, jj])
            plt.imshow(np.log(np.abs(psf_[n-28:n+28,n-28:n+28])))#[128-32:128+32,128-32:128+32])))
            if jj % 4==0:
                plt.ylabel('%0.1fx' % rr)
            if jj < 4 and ii==0:
                plt.title(r'$\gamma=$%0.1f' % nn)
            plt.xticks([])
            plt.yticks([])
            kk+=1

            lr = hr2lr.convolvehr(hr, psf_, nbit=16, rebin=4)
            dsr = (resolve_single(model,lr)).numpy()
            ssim = tf.image.ssim(dsr[None, ...,0, None].astype(np.uint16), 
                                 hr[None, ..., None].astype(np.uint16), 
                                 2**(nbit)-1, filter_size=2, 
                                 filter_sigma=1.5, k1=0.01, k2=0.03)
            psnr_ = tf.image.psnr(dsr[None, ...,0,None].astype(np.uint16),
                          hr[None, ..., None].astype(np.uint16),
                          max_val=2**(nbit)-1).numpy()[0]
            psnr_arr.append(psnr_)
            ssim_arr.append(ssim)

            plt.subplot(gs[ii, jj+3])
            plt.imshow(dsr[200:400,400:600,0], vmax=hr.max()*0.01)
            plt.axis('off')

    psnr_arr = np.array(psnr_arr).reshape(3,3)
    ssim_arr = np.array(ssim_arr).reshape(3,3)

    ax = plt.subplot(gs[:3, 6:9])
    immy = ax.imshow(psnr_arr, cmap='RdBu')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(immy,cax=cax,label='PSNR')

        # psnr_ = tf.image.psnr(dsr[None, ...,0,None].astype(np.uint16),
        #               hr[None, ..., None].astype(np.uint16),
        #               max_val=2**(nbit)-1).numpy()[0]
def restore_CLEAN(fnfits_model, bmaj=3.56, 
                  bmin=3.51818, pa=-15.0404, pixel_scale=0.5, fnout=None):
    """ 
    fnfits_model : CLEAN components fits file
    bmaj : arcsec
    bmin : arcsec 
    pa : degrees
    """
    f = fits.open(fnfits_model)
    data = f[0].data[0,0]
    header = f[0].header
    try:
        pixel_scale = abs(f[0].header['CDELT1'])*3600
    except:
        pixel_scale = pixel_scale
    print("Using %0.2f" % pixel_scale)

    sigma_x = (bmaj+bmin)/2.355 / pixel_scale
    sigma_y = (bmaj+bmin)/2.355 / pixel_scale
    data_restored = gaussian_filter(data, (sigma_x, sigma_y))

    if fnout is not None:
        hdu = fits.PrimaryHDU(data_restored, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fnout)

    return data_restored


def gaussian2D(coords,  # x and y coordinates for each image.
                  amplitude=1,  # Highest intensity in image.
                  xo=0,  # x-coordinate of peak centre.
                  yo=0,  # y-coordinate of peak centre.
                  sigma_x=1,  # Standard deviation in x.
                  sigma_y=1,  # Standard deviation in y.
                  rho=0,  # Correlation coefficient.
                  offset=0,
                  rot=0):  # rotation in degrees.
    x, y = coords

    rot = np.deg2rad(rot)

    x_ = np.cos(rot)*x - y*np.sin(rot)
    y_ = np.sin(rot)*x + np.cos(rot)*y

    xo = float(xo)
    yo = float(yo)

    xo_ = np.cos(rot)*xo - yo*np.sin(rot) 
    yo_ = np.sin(rot)*xo + np.cos(rot)*yo

    x,y,xo,yo = x_,y_,xo_,yo_

    # Create covariance matrix
    mat_cov = [[sigma_x**2, rho * sigma_x * sigma_y],
               [rho * sigma_x * sigma_y, sigma_y**2]]
    mat_cov = np.asarray(mat_cov)
    # Find its inverse
    mat_cov_inv = np.linalg.inv(mat_cov)

    # PB We stack the coordinates along the last axis
    mat_coords = np.stack((x - xo, y - yo), axis=-1)

    G = amplitude * np.exp(-0.5*np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                    mat_cov_inv),
                                          mat_coords[..., np.newaxis])) + offset
    return G.squeeze()

def lobe_gal_plot(model=None):
    if model is None:
        lr = np.load('./plots/ska-fun-mid-dirty.npy')
        sr = np.load('./plots/ska-fun-mid-SR.npy')
        hr = np.load('./plots/ska-fun-mid-true.npy')
        hr = hr2lr.normalize_data(hr)
    else:
        pass


    et = [0,1,1,0]

    figure()
    subplot(131)
    imshow(lr, cmap='afmhot',extent=et, vmax=lr.max()*.13)
    xlim(0.2, 0.8)
    ylim(0.8, 0.2)
    axis('off')
    title('Dirty image', c='C1')

    subplot(132)
    imshow(sr, cmap='afmhot', vmax=sr.max()*0.1, extent=et)
    xlim(0.2, 0.8)
    ylim(0.8, 0.2)
    title('POLISH reconstruction', c='C2')
    axis('off')
    
    subplot(133)
    imshow(hr, vmax=hr.max()*0.05, extent=et, cmap='afmhot')
    xlim(0.2, 0.8)
    ylim(0.8, 0.2)
    axis('off')
    title('True sky', c='k')
    tight_layout()

    figure()
    subplot(131)
    imshow(lr, cmap='afmhot',extent=et, vmax=lr.max()*.15)
    xlim(0.46, 0.52)
    ylim(0.55, 0.46)
    axis('off')

    subplot(132)
    imshow(sr, cmap='afmhot', vmax=sr.max()*0.11, extent=et)
    xlim(0.46, 0.52)
    ylim(0.55, 0.46)
    axis('off')
    
    subplot(133)
    imshow(hr, vmax=hr.max()*0.05, extent=et, cmap='afmhot')
    xlim(0.46, 0.52)
    ylim(0.55, 0.46)
    axis('off')
    tight_layout()



def vla_mosaic():
    d = np.load('plots/vla-dirty-image.npy')
    d = hr2lr.normalize_data(d)
    arr = np.zeros([8192, 8192])
    for ii in range(4):
        for jj in range(4):
            dsr = resolve_single(model, d[1024*ii:1024*(ii+1), 1024*jj:1024*(jj+1)]).numpy()
            arr[2*1024*ii:2*1024*(ii+1), 2*1024*jj:2*1024*(jj+1)] = dsr


def plot_vla_polish(cleanmin=200, polishmin=-800):
    lr = np.load('./plots/vla-dirty-plotregion.npy')
    cr = np.load('./plots/vla-CLEAN10k-plotregion.npy')
    sr = np.load('./plots/vla-polish-plotregion.npy')

    cr = hr2lr.normalize_data(cr)
    lr = hr2lr.normalize_data(lr)

    et = [0,1,1,0]

    #figure()
    fig, ax = plt.subplots()
    plt.subplot(131)
    plt.imshow(lr, cmap='afmhot',extent=et, vmax=lr.max()*.075)
#    rect = patches.Rectangle((0.28, 0.09), .15, 0.2,  linewidth=3, edgecolor='C3', facecolor='none')
#    ax.add_patch(rect)
    axis('off')
    title('Dirty image', c='C3', fontsize=28)
    xlim(0.13, .77)
    ylim(0.85, 0.1)

    subplot(132)
    imshow(sr, cmap='afmhot', vmax=sr.max()*0.025, extent=et, vmin=polishmin)
    title('POLISH reconstruction', c='C2', fontsize=28)
    axis('off')
    xlim(0.13, .77)
    ylim(0.85, 0.1)
    
    subplot(133)
    imshow(cr, vmax=cr.max()*0.08, extent=et, cmap='afmhot', vmin=cleanmin)
    xlim(0.13, .77)
    ylim(0.85, 0.1)
    axis('off')
    title('CLEAN', c='C0', fontsize=28)
    tight_layout()

    figure()
    subplot(131)
    imshow(lr, cmap='afmhot',extent=et, vmax=lr.max()*.08)
    xlim(0.28, 0.28+.15)
    ylim(0.09+0.2, 0.09)
    axis('off')

    subplot(132)
    imshow(sr, cmap='afmhot', vmax=sr.max()*0.025, extent=et, vmin=polishmin)
    xlim(0.28, 0.28+.15)
    ylim(0.09+0.2, 0.09)
    axis('off')
    
    subplot(133)
    imshow(cr, vmax=cr.max()*0.08, extent=et, cmap='afmhot',vmin=cleanmin)
    xlim(0.28, 0.28+.15)
    ylim(0.09+0.2, 0.09)
    axis('off')
    tight_layout()




def create_fits_header():
    hdu = fits.PrimaryHDU()
    hdr = hdu.header 
    hdr['BITPIX'] = -32
    hdr['NAXIS'] = 4
    hdr['NAXIS1'] = self._nx 
    hdr['NAXIS2'] = self._ny 
    hdr['NAXIS3'] = self._nchan
    hdr['NAXIS4'] = 1
    hdr['OBJECT']  = 'source  '                                                            
    hdr['CTYPE1']  = 'RA---SIN'
    hdr['CRPIX1']  = self._nx//2
    hdr['CRVAL1']  = 0.
    hdr['CDELT1']  = -self._pixel_size/3600 
    hdr['CUNIT1']  = 'deg     '                                                            
    hdr['CTYPE2']  = 'DEC--SIN'
    hdr['CRPIX2']  = self._ny//2                                                  
    hdr['CRVAL2']  = 37.1298333333333                                                  
    hdr['CDELT2']  = self._pixel_size/3600                                                   
    hdr['CUNIT2']  = 'deg     '                                                            
    hdr['CTYPE3']  = 'FREQ    '
    hdr['CRPIX3']  = 1.                                                  
    hdr['CRVAL3']  = 1e9*(0.5*(self._freqmin+self._freqmax))
    hdr['CDELT3']  = self._delta_freq
    hdr['CUNIT3']  = 'Hz      '
    hdr['CTYPE4']  = 'STOKES  '                                                            
    hdr['CRPIX4']  =  1.                                                  
    hdr['CRVAL4']  =  1.                                                  
    hdr['CDELT4']  =  1.                                                  
    hdr['CUNIT4']  = '        '
    self.fits_header = hdr
    return hdr 

def write_data_fits(data, header, fnout):
    hdu = fits.PrimaryHDU(data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fnout)




import sys 

import numpy as np
import matplotlib.pylab as plt
import glob
from astropy.io import fits

from astropy.coordinates import SkyCoord
from astropy import units as u

def readcat(fncat):
    #   1 NUMBER          Running object number
    #   2 FLUX_AUTO       Flux within a Kron-like elliptical aperture     [count]
    #   3 FLUXERR_AUTO    RMS error for AUTO flux                         [count]
    #   4 X_IMAGE         Object position along x                         [pixel]
    #   5 Y_IMAGE         Object position along y                         [pixel]
    #   6 A_IMAGE         Profile RMS along major axis                    [pixel]
    #   7 B_IMAGE         Profile RMS along minor axis                    [pixel]
    #   8 THETA_IMAGE     Position angle (CCW/x)                          [deg]
    #   9 ELONGATION      A_IMAGE/B_IMAGE
    #  10 ELLIPTICITY     1 - B_IMAGE/A_IMAGE
    #  11 FLAGS           Extraction flags
    param_arr = np.genfromtxt(fncat)
    number = param_arr[:,0]
    flux_auto = param_arr[:,1]
    flux_err_auto = param_arr[:,2]
    x_image = param_arr[:,3]
    y_image = param_arr[:,4]
    A_image = param_arr[:,5]
    B_image = param_arr[:,6]
    theta_image = param_arr[:,7]
    el = param_arr[:,9]

    return number, flux_auto, x_image, y_image, A_image, B_image, theta_image, el, flux_err_auto

def plot_sky(fn, fncat, sizevar='flux'):
    param_arr = readcat(fncat)
    print(len(param_arr))
    number, flux_auto, x_image, y_image, A_image, B_image, theta_image, el = param_arr
    if fn.endswith('.fits'):
        data = fits.open(fn)
        data = data[0].data
    elif fn.endswith('.npy'):
        data = np.load(fn)
    elif fn.endswith('.png'):
        data = load_image(fn)
    else:
        print("Expected fits, npy, or png.")
        return
    plt.figure()
    plt.imshow(data, vmax=data.max()*0.025)

    if sizevar=='flux':
        plt.scatter(x_image, y_image, flux_auto/flux_auto.max()*100, 
                marker='o',edgecolor='red',facecolors='none')
    elif sizevar=='A_image':
        plt.scatter(x_image, y_image, A_image/A_image.max()*30, 
                marker='o',edgecolor='red',facecolors='none')


#    plt.show()
    return data, param_arr

def plot_comparison(param_arr_l):
    plt.figure(figsize=(7,7))
    # Loop through True, SR, LR
    alph = [1, 1, 0.75]
    colors = ['C0', 'C1', 'C3']
    for ii in [1,0,2]:
        number, flux_auto, x_image, y_image, A_image, B_image, theta_image, el = param_arr_l[ii]
        plt.subplot(221)
        plt.hist(A_image, bins=200, log=True, alpha=alph[ii], range=(0,50), density=True, color=colors[ii])
        plt.xlim(-0.5,15)
        plt.ylabel('fraction')
        plt.xlabel('semi-major (pixels)')
        plt.subplot(222)
        plt.hist(B_image, bins=200, log=True, alpha=alph[ii], range=(0,50), density=True,color=colors[ii])
        plt.xlim(-0.5,10)
        plt.xlabel('semi-minor (pixels)')
        plt.subplot(223)
        plt.hist(flux_auto, bins=200, log=True, alpha=alph[ii], density=True,color=colors[ii])
        plt.xlabel('flux')
        plt.ylabel('fraction')
        plt.subplot(224)
        plt.hist(el, bins=200, log=True, alpha=alph[ii], density=True,color=colors[ii])
        plt.xlabel('elongation')
    plt.subplot(223)
    plt.legend(['True', 'NN Recon', 'Dirty'])
    plt.show()

def gather(fnhr, fnsr, fnlr=None):
    number_sr, flux_auto_sr, x_image_sr, y_image_sr, A_image_sr, B_image_sr, theta_image_sr, el_sr = readcat(fnsr)
    number_hr, flux_auto_hr, x_image_hr, y_image_hr, A_image_hr, B_image_hr, theta_image_hr, el_hr = readcat(fnhr)

    if fnlr is not None:
        number_lr, flux_auto_lr, x_image_lr, y_image_lr, A_image_lr, B_image_lr, theta_image_lr, el_lr = readcat(fnlr)
        x_image_lr += 61.
        y_image_lr += 61
        indarr_lr = []

    param_arr_all = []
    indarr_hr = []

    if fnlr is None:
        for ii in range(x_image_hr.shape[0]):
            r_sr = np.sqrt((y_image_hr[ii]-y_image_sr)**2 + (y_image_hr[ii]-x_image_sr)**2)
            ind_sr = np.argmin(r_sr)
            indarr_hr.append(ind_sr)
            if r_sr.min()<10.0:
                param_arr_all.append([A_image_hr[ii], B_image_hr[ii], flux_auto_hr[ii], 
                                      A_image_sr[ind_sr], B_image_sr[ind_sr], flux_auto_sr[ind_sr]])

        param_arr_all = np.concatenate(param_arr_all).reshape(-1, 6)

    elif fnlr is not None:
        for ii in range(x_image_hr.shape[0]):
            r_hr = np.sqrt((y_image_hr[ii]-y_image_sr)**2 + (y_image_hr[ii]-x_image_sr)**2)
            if fnlr is not None:
                r_lr = np.sqrt((y_image_hr[ii]-y_image_lr)**2 + (y_image_hr[ii]-x_image_lr)**2)
            ind_hr = np.argmin(r_hr)
            ind_lr = np.argmin(r_lr)
            indarr_lr.append(ind_hr)
            indarr_hr.append(ind_lr)
            print(ii, r_hr.min(), r_lr.min())

            if r_hr.min()<25.0:# and r_lr.min()<5.0:
                print(ii)
                param_arr_all.append([A_image_hr[ii], B_image_hr[ii], flux_auto_hr[ii], 
                                      A_image_sr[ind_hr], B_image_sr[ind_hr], flux_auto_sr[ind_hr], 
                                      A_image_lr[ind_lr], B_image_lr[ind_lr], flux_auto_lr[ind_lr],])

        param_arr_all = np.concatenate(param_arr_all).reshape(-1, 9)

    return param_arr_all
#         # ax1.errorbar([x_image[ii]], [y_image[ii]], yerr=[0.25], fmt="o", color="black", ms=0.1, zorder=1)
#         # ax1.add_artist(Ellipse((x_image[ii], y_image[ii]), A_image[ii], B_image[ii], angle=0*theta_image[ii], facecolor="C1", edgecolor="C1",zorder=2))
#         # ax1.errorbar([x_imagesr[ind]], [y_imagesr[ind]], yerr=[0.25], fmt="o", color="C2", ms=0.1, zorder=1)
#         # ax1.add_artist(Ellipse((x_imagesr[ind], y_imagesr[ind]), A_imagesr[ind], B_imagesr[ind], angle=0*theta_imagesr[ind], facecolor="C0",edgecolor="C0",zorder=2,alpha=0.75))

def match(fn1, fn2, pixel_scale_arcsec=0.5):
    p1 = readcat(fn1)
    number_1, flux_auto_1, x_image_1, y_image_1, A_image_1, B_image_1, theta_image_1, el_1, flux_auto_err_1 = p1
    p2 = readcat(fn2)
    number_2, flux_auto_2, x_image_2, y_image_2, A_image_2, B_image_2, theta_image_2, el_2, flux_auto_err_2 = p2

    pixel_scale = pixel_scale_arcsec / 3600.

    ra1, dec1 = x_image_1 * pixel_scale, y_image_1 * pixel_scale
    ra2, dec2 = x_image_2 * pixel_scale, y_image_2 * pixel_scale

    c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)

    return ra1, dec1, ra2, dec2, p1, p2, idx

def plot_all(fn1, fn2, fn3=None, pixel_scale_arcsec=0.5, 
             psf='AJ-15x60s-4000chan-0.5arcsec-3x/psf/psf.npy'):
    
    fn1 = 'plots/0808.cat'
    fn2 = 'plots/0808SR.cat'
    fn3 = 'plots/0808CLEAN-50k-maj.cat'

    x = np.linspace(0,8,int(8.0/pixel_scale_arcsec))

    if psf is not None:
        from scipy.interpolate import interp1d
        psf = np.load(psf)
        psf = psf[512, 512:512+16]
        f = interp1d(x, psf, kind='cubic')

    ra1, dec1, ra2, dec2, p1, p2, idx = match(fn1, fn2, pixel_scale_arcsec=pixel_scale_arcsec)

    number_1, flux_auto_1, x_image_1, y_image_1, A_image_1, B_image_1, theta_image_1, el_1, flux_auto_err_1 = p1
    number_2, flux_auto_2, x_image_2, y_image_2, A_image_2, B_image_2, theta_image_2, el_2, flux_auto_err_2 = p2

    if fn3 is not None:
        ra1, dec1, ra3, dec3, p1, p3, idx3 = match(fn1, fn3, pixel_scale_arcsec=pixel_scale_arcsec)        
        number_3, flux_auto_3, x_image_3, y_image_3, A_image_3, B_image_3, theta_image_3, el_3, flux_auto_err_3 = p3

    snr_2 = flux_auto_2/flux_auto_err_2
    snr_3 = flux_auto_3/flux_auto_err_3

    figure(figsize=(7,5))

    subplot(121)
    scatter(A_image_1*pixel_scale_arcsec, A_image_2[idx]*pixel_scale_arcsec, np.log(snr_2[idx]), c = '#95D840FF', alpha=0.75)
#    scatter(A_image_1*pixel_scale_arcsec, A_image_2[idx]*pixel_scale_arcsec, np.log10(snr_2[idx]), color=c, alpha=0.75)
    xlim(0.5, 7)
    xlabel(r"True  $\theta_A$  (arcseconds) ")
    ylabel(r"$\hat{\theta}_A$  (arcseconds)")
    ylim(0, 7.5)

    subplot(122)
    scatter(B_image_1*pixel_scale_arcsec, B_image_2[idx]*pixel_scale_arcsec, np.log(snr_2[idx]), c = '#95D840FF', alpha=0.85)
    xlabel(r"True  $\theta_B$  (arcseconds) ")
    ylabel(r"$\hat{\theta}_B$  (arcseconds)")
    xlim(0.5, 7)
    ylim(0, 7.5)


    if fn3 is not None:
        subplot(121)
        scatter(A_image_1*pixel_scale_arcsec, A_image_3[idx3]*pixel_scale_arcsec, np.log(snr_2[idx]), c = 'k', alpha=0.5)
    #    scatter(A_image_1*pixel_scale_arcsec, A_image_2[idx]*pixel_scale_arcsec, np.log10(snr_2[idx]), color=c, alpha=0.75)
        xlim(0.5, 7)
        ylim(0, 7.5)

        if psf is not None:
            xx = np.linspace(0, 8, 100)
            plot(xx, f(xx)*6.5, alpha=0.25, lw=2, c='C3')

        subplot(122)
        scatter(B_image_1*pixel_scale_arcsec, B_image_3[idx3]*pixel_scale_arcsec, np.log(snr_2[idx]), c='k', alpha=0.5)

        if psf is not None:
            plot(xx, f(xx)*6.5, alpha=0.25, lw=2, c='C3')

        xlim(0.5, 7)
        ylim(0, 7.5)

        legend(['PSF/kernel', 'POLISH', 'CLEAN'], loc=1)

    subplot(121)
    plot(x, x, '--', alpha=0.6)
    subplot(122)
    plot(x, x, '--', alpha=0.6)

    tight_layout()

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
































