import os
import sys

import matplotlib
import numpy as np
import matplotlib.pylab as plt
import optparse
from scipy import signal
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from model import resolve_single
from model.edsr import edsr
from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.common import tf#resolve_single16, tf
import hr2lr

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

PIXEL_SIZE = 0.25 # resolution of HR map in arcseconds

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


def plotter(datalr, datasr, datahr=None, dataother=None,
            cmap='viridis', suptitle=None, 
            fnfigout='test.pdf', vm=None, nbit=16, calcpsnr=True):

    fig=plt.figure(figsize=(10,7.8))

    datasr = datasr.numpy()

    datalr = hr2lr.normalize_data(datalr, nbit=nbit)
    # datalr = datalr - np.median(datalr)
    #datasr = hr2lr.normalize_data(datasr, nbit=nbit)
    #datahr = hr2lr.normalize_data(datahr, nbit=nbit)

    if datahr is None:
        nsub=2
    else:
        nsub=3
    if datahr is not None:
      pass

    if dataother is not None:
        nsub += 1

    if calcpsnr:
        psnr_ = tf.image.psnr(datasr[None, ..., 0, None].astype(np.uint16), 
                             datahr[None, ..., None].astype(np.uint16), 
                             max_val=2**(nbit)-1)
        ssim = tf.image.ssim(datasr[None, ..., 0, None].astype(np.uint16), 
                             datahr[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr = "PSNR = %0.1f\nSSIM = %0.4f" % (psnr_, ssim)

#         if nsub==4:
#             print(dataother.shape, datahr.shape)
# #            dataother = hr2lr.normalize_data(dataother, nbit=nbit)
#             psnr_clean = tf.image.psnr(dataother[None, ..., None].astype(np.uint16), 
#                                  datahr[None, ..., None].astype(np.uint16), 
#                                  max_val=2**(nbit)-1)
#             ssim_clean = tf.image.ssim(dataother[None, ..., None].astype(np.uint16), 
#                                  datahr[None, ..., None].astype(np.uint16), 
#                                  2**(nbit)-1, filter_size=2, 
#                                  filter_sigma=1.5, k1=0.01, k2=0.03)
#             psnr_clean_ = "PSNR = %0.1f\nSSIM = %0.4f" % (psnr_clean, ssim_clean)


    # np.save('lr',datalr)
    # np.save('hr',datahr)
    # np.save('sr',datasr)

    if vm is None:
      vminlr=max(0.9*np.median(datalr), 0)
      vmaxlr=np.median(datalr)+0.05*(np.max(datalr)-np.median(datalr))

      vminsr=max(0.9*np.median(datasr), 0)
      vmaxsr=np.median(datasr)+0.05*(np.max(datasr)-np.median(datasr))

      vminhr=max(0.9*np.median(datahr), 0)
      vmaxhr=np.median(datalr)+0.05*(np.max(datalr)-np.median(datalr))
    else:
      vminlr, vminsr, vminhr = 0, 0, 0
      vmaxlr, vmaxsr, vmaxhr = vm, vm, vm

    vmaxlr=22500
    vminlr=0
    vmaxsr=vm
    vmaxhr=vm

    ax1 = plt.subplot(2,nsub,1)
    plt.title('Dirty map', color='C1', fontweight='bold', fontsize=15)
    plt.axis('off')
    plt.imshow(datalr[...,0], cmap=cmap, vmax=vmaxlr, vmin=vminlr, 
               aspect='auto', extent=[0,1,0,1])
    plt.setp(ax1.spines.values(), color='C1')

    ax2 = plt.subplot(2,nsub,2, sharex=ax1, sharey=ax1)
    plt.title('NN reconstruction', color='C2', 
              fontweight='bold', fontsize=15)
    plt.imshow(datasr[...,0], cmap=cmap, vmax=vmaxsr, vmin=vminsr, 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')

    if calcpsnr:
      print("PSNR")
      plt.text(0.6, 0.85, psnr, color='white', fontsize=7, fontweight='bold')

    if nsub>=3:
        ax5 = plt.subplot(2,nsub,3,sharex=ax1, sharey=ax1)
        plt.title('True map', color='k', fontweight='bold', fontsize=15)
        plt.imshow(datahr, cmap=cmap, vmax=vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])
        plt.axis('off')

    if nsub==4:
        ax55 = plt.subplot(2,nsub,4,sharex=ax1, sharey=ax1)
        plt.title('CLEAN', color='k', fontweight='bold', fontsize=15)
        plt.imshow(dataother, cmap=cmap, vmax=200, vmin=0, aspect='auto', extent=[0,1,0,1])
        plt.axis('off')

        #if calcpsnr:
        #    plt.text(0.6, 0.85, psnr_clean_, color='white', fontsize=7, fontweight='bold')
            

    ax3 = plt.subplot(2,nsub,nsub+1)
    plt.axis('off')
    plt.xlim(0.25,0.45)
    plt.ylim(0.25,0.45)
    plt.imshow(datalr[:,:,0], cmap=cmap, vmax=vmaxlr, vmin=vminlr, 
              aspect='auto', extent=[0,1,0,1])
    plt.title('Dirty map \nzoom', color='C1', fontweight='bold', fontsize=15)

    ax4 = plt.subplot(2,nsub,nsub+2,sharex=ax3, sharey=ax3)
    plt.title('NN reconstruction\nzoom ', color='C2', 
              fontweight='bold', fontsize=15)

    plt.imshow(datasr[:,:,0], cmap=cmap, 
              vmax=vmaxsr, vmin=vminsr, aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.xlim(0.25,0.45)
    plt.ylim(0.25,0.45)
    plt.suptitle(suptitle, color='C0', fontsize=20)

    if nsub >=3:
        ax6 = plt.subplot(2,nsub,nsub+3,sharex=ax3, sharey=ax3)
        plt.title('True map', color='k', fontweight='bold', fontsize=15)        
        plt.imshow(datahr[:,:], cmap=cmap, 
                   vmax=vmaxhr, vmin=vminhr, aspect='auto', extent=[0,1,0,1])
        plt.xlim(0.25,0.45)
        plt.ylim(0.25,0.45)
        plt.axis('off')
    if nsub==4:
        ax8 = plt.subplot(2,nsub,nsub+4,sharex=ax3, sharey=ax3)
        plt.title('CLEAN', color='k', fontweight='bold', fontsize=15)        
        plt.imshow(dataother, cmap=cmap, 
                   vmax=100, vmin=0.0, aspect='auto', extent=[0,1,0,1])
        plt.xlim(0.25,0.45)
        plt.ylim(0.25,0.45)
        plt.axis('off')
    else:
        plt.axis('off')

    plt.show()

def func(fn_img, fn_model, psf=None, fnother=None,
         fn_img_hr=None, suptitle=None, 
         fnfigout='test.pdf', vm=75, scale=4,
         nbit=8, distortpsf=False,ksize=64, 
         alphad=0, fitgal=None):

    if fn_img.endswith('npy'):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith('png'):
      try:
          datalr = load_image(fn_img)
      except:
          datalr = load_image('demo/0851x4-crop.png')

    elif fn_img.endswith('.fits'):
        f = fits.open(fn_img)
        datalr = f[0].data[0,0]
    else:
      print('Do not recognize input image file type, exiting')
      exit()


    if fn_img_hr!=None:
        if fn_img_hr.endswith('.npy'):
            datahr = np.load(fn_img_hr)
        elif fn_img_hr.endswith('png'):
            datahr = load_image(fn_img_hr)
    else:
        datahr = None

    if psf is not None:
        if datahr is None:
          pass
        print("Convolving data")
        if psf in ('gaussian','Gaussian'):
          kernel1D = signal.gaussian(8, std=1).reshape(8, 1)
          kernel = np.outer(kernel1D, kernel1D)
        elif psf.endswith('.npy'):
          kernel = np.load(psf)
          nkern = len(kernel)
          print(kernel.shape)
          kernel = kernel[nkern//2-ksize:nkern//2+ksize, nkern//2-ksize:nkern//2+ksize]
          print(kernel.shape)
        else:
          print("Can't interpret kernel")
          exit()

        if distortpsf:
          plt.figure()
          plt.subplot(121)
          plt.imshow(kernel)
          plt.title('Original')

#          alpha = np.random.randint(0,20)
          kernel = elastic_transform(kernel[...,None]*np.ones([1,1,3]), 
                                     alpha=alphad, 
                                     sigma=2, alpha_affine=0)[..., 0]
          plt.subplot(122)
          plt.imshow(kernel)
          plt.title('Distorted')
          plt.show()
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(datalr, vmax=25000)
        plt.subplot(133)
        plt.hist(datalr.flatten(), bins=100, log=True)
        datalr = hr2lr.convolvehr(datahr, kernel, plotit=False, 
                            rebin=4, norm=True, nbit=nbit)
#        datalr = hr2lr.convolvehr(datahr, kernel, rebin=4)
#        datalr = hr2lr.normalize_data(datalr, nbit=nbit)
        plt.subplot(132)
        plt.imshow(datalr, vmax=25000)
        plt.subplot(133)
        plt.hist(datalr.flatten(), bins=100, log=True, alpha=0.25)
        plt.show()
    else:
        print("Assuming data is already convolved")

    model = wdsr_b(scale=scale, num_res_blocks=32)
    model.load_weights(fn_model)

    datalr = datalr[:,:,None]
#    datalr += np.random.normal(0, 0.001*datalr.max(), datalr.shape).astype(datalr.dtype)
#    datalr = hr2lr.normalize_data(datalr, nbit=nbit)

    datasr = resolve_single(model, datalr, nbit=nbit)
    print(datasr.shape, datalr.shape)
    exit()
    if fitgal:
      import scipy.optimize as opt
      galparams = np.genfromtxt(fitgal)
      ngal = len(galparams)
      
      for jj in range(ngal):
        
        xind, yind, sigx, sigy, rho, flux = galparams[jj]
        xind, yind = int(xind), int(yind)
        sig2 = int(max(5*sigx, 5*sigy))
        data2fit = datasr[xind-sig2:xind+sig2, yind-sig2:yind+sig2,0].numpy()
        nx, ny = data2fit.shape
        coords = np.meshgrid(np.arange(nx), np.arange(ny))
        # params:  amplitude=1 xo yo sigma_x sigma_y rho offset
        initial_guess = [data2fit.max(), nx//2, ny//2, sigx, sigy, rho, np.median(data2fit)]
        try:
          popt, pcov = opt.curve_fit(hr2lr.Gaussian2D_v1_flatten, coords, data2fit.ravel(), p0=initial_guess)
          datafit = hr2lr.Gaussian2D_v1(coords, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
          sigxopt, sigyopt = popt[3], popt[4]
        except RuntimeError:
          print("Didn't converge")

        plt.figure(figsize=(10,6))
        plt.subplot(131)
        plt.imshow(datahr[xind-sig2:xind+sig2, yind-sig2:yind+sig2], aspect='auto', extent=[0,1,0,1])
        plt.text(0.62, 0.8, r'$\sigma_x=%0.1f$'%abs(sigy), color='w', fontsize=8)
        plt.text(0.62, 0.75, r'$\sigma_y=%0.1f$'%abs(sigx), color='w', fontsize=8)
        plt.title('True galaxy')
        plt.subplot(132)
        plt.imshow(data2fit, aspect='auto', extent=[0,1,0,1])
        plt.title('NN reconstruction', c='green')
        plt.subplot(133)
        plt.imshow(datafit,aspect='auto', extent=[0,1,0,1])
        plt.title('2D Gaussian fit', color='k', alpha=0.5)
        plt.text(0.62, 0.8, r'$\sigma_{x fit}=%0.1f$'%abs(sigxopt), color='w', fontsize=8)
        plt.text(0.62, 0.75, r'$\sigma_{y fit}=%0.1f$'%abs(sigyopt), color='w', fontsize=8)
        plt.text(0.62, 0.7, r'$ratio_{x}=%0.2f$'%abs(sigyopt/sigx), color='w', fontsize=8)
        plt.text(0.62, 0.65, r'$ratio_{x}=%0.2f$'%abs(sigxopt/sigy), color='w', fontsize=8)
        plt.show()
        print(xind, yind, sigx, sigy, flux)
        print(popt)

    if fnother is not None:
        if fnother.endswith('fits'):
            ff = fits.open(fnother)
            dataclean = ff[0].data[0,0]
            dataclean = gaussian_filter(dataclean, sigma=4)
    else:
        dataclean = None
            #dataclean = hr2lr.normalize_data(dataclean, nbit=16)

    plotter(datalr, datasr, datahr=datahr, dataother=dataclean,
            suptitle=suptitle, fnfigout=fnfigout, vm=vm, 
            nbit=nbit)

if __name__=='__main__':
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = optparse.OptionParser(prog="hr2lr.py",
                                   version="",
                                   usage="%prog image weights.h5  [OPTIONS]",
                                   description="Take high resolution images, convolve them, \
                                   and save output.")

    parser.add_option('-f', dest='fnhr', 
                      help="high-res file name", default=None)
    parser.add_option('-k', '--psf', dest='psf', type='str',
                      help="If None, assume image is already low res", default=None)
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=64)
    parser.add_option('-t', '--title', dest='title', type='str',
                      help="Super title for plot", default=None)
    parser.add_option('-o', '--fnfigout', dest='fnfigout', default='test.pdf')    
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits in image", default=8)    
    parser.add_option('-a', '--alpha', dest='alphad', type=float,
                      help="affine distortion parameter", default=0)
    parser.add_option('--vm', dest='vm', 
                      help="vmax in imshow figure", default=None)
    parser.add_option('--distort-psf', dest='distortpsf', 
                      help="alter psf randomly", action="store_true")
    parser.add_option('--fit', dest='fitgal', type=str, default=None, 
                      help="fit to list of galaxies in txt file", )
    parser.add_option('-c', '--compare', dest='fncompare', type=str, default=None, 
                      help="e.g. CLEANd data", )

    options, args = parser.parse_args()
    fn_img, fn_model = args
    
    assert os.path.exists(fn_img)
    assert os.path.exists(fn_model)

    if options.fncompare is not None:
        assert os.path.exists(options.fncompare)

    func(fn_img, fn_model, psf=options.psf, 
         fn_img_hr=options.fnhr, fnother=options.fncompare,
         suptitle=options.title, 
         fnfigout=options.fnfigout,
         vm=options.vm, nbit=options.nbit,
         distortpsf=options.distortpsf, 
         ksize=options.ksize, alphad=options.alphad, 
         fitgal=options.fitgal, scale=options.rebin)





