import sys, os

import matplotlib.pylab as plt
import numpy as np
import glob
import cv2
from scipy import signal, interpolate
import optparse

from astropy.modeling.models import Sersic2D
import simulation

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

try:
    from astropy.io import fits 
except:
    print("Could not load astropy.io.fits")

PIXEL_SIZE = 0.25 # resolution of HR map in arcseconds
fn_background = './data/haslam-cdelt0.031074-allsky.npy'
src_density = 5 # per sq arcminute 
NSIDE = 2304 # number of pixels per side for high res image
FREQMIN, FREQMAX = 0.7, 2.0

def readfits(fnfits):
    hdulist = fits.open(fnfits)
    data = hdulist[0].data[0, 0]
    header = hdulist[0].header
    pixel_scale = abs(header['CDELT1'])
    num_pix = abs(header['NAXIS1'])
    return data, header, pixel_scale, num_pix


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
                                          mat_cofords[..., np.newaxis])) + offset
    return G.squeeze()

def normalize_data(data, nbit=16):
    data = data - data.min()
    data = data/data.max()
    data *= (2**nbit-1)
    if nbit==16:
        data = data.astype(np.uint16)
    elif nbit==8:
        data = data.astype(np.uint8)
    return data

def convolvehr(data, kernel, plotit=False, 
               rebin=4, norm=True, nbit=16, noise=True):
    if len(data.shape)==3:
        kernel = kernel[..., None]
        ncolor = 1
    else:
        ncolor = 3
    
    if noise:
#        dataLR += 10*np.random.chisquare(5,dataLR.shape)
        data_noise = data + np.random.normal(0,5,data.shape)
    else:
        data_noise = data

    dataLR = signal.fftconvolve(data_noise, kernel, mode='same')

    if norm is True:
         dataLR = normalize_data(dataLR, nbit=nbit)
         data = normalize_data(data, nbit=nbit)

    dataLR = dataLR[rebin//2::rebin, rebin//2::rebin]

    if plotit:
        plt.figure()
        dataLRflat = dataLR.flatten()
        dataLRflat = dataLRflat[dataLRflat!=0]
        dataflat = data.flatten()
        dataflat = dataflat[dataflat!=0]
        plt.hist(dataLRflat, color='C1', alpha=0.5, 
                 density=True, log=True, bins=255)
        plt.hist(dataflat, bins=255, color='C0', alpha=0.25, 
                 density=True, log=True)
        plt.title('Bit value distribution', fontsize=20)
        plt.xlabel('Pixel value')
        plt.ylabel('Number of pixels')
        plt.legend(['Convolved','True'])
        plt.figure()
        if norm is False:
            data = data.reshape(data.shape[0]//4,4,
                                data.shape[-2]//4, 4, 
                                ncolor).mean(1).mean(-2)
            plt.imshow(dataLR[..., 0], cmap='Greys', 
                        vmax=dataLR[..., 0].max()*0.025)
        else:
            plt.imshow(dataLR, vmax=dataLR[..., 0].max())
        plt.title('Convolved', fontsize=15)
        plt.figure()
        if norm is False:
            plt.imshow(data[..., 0], cmap='Greys', vmax=data.max()*0.1)
        else:
            plt.imshow(data, vmax=data.max()*0.1)
        plt.title('True', fontsize=15)
        plt.figure()
        plt.imshow(kernel[...,0])
        plt.title('Kernel / PSF', fontsize=20)
        plt.show()

    return dataLR, data_noise

def create_LR_image(fl, kernel, fdirout=None, 
                    pointsrcs=False, plotit=False, 
                    norm=True, sky=False, rebin=4, nbit=8, 
                    distort_psf=False,
                    nimages=800, nchan=1, save_img=True):
    if type(fl) is str:
        fl = glob.glob(fl+'/*.png')
    elif type(fl) is list:
        pass
    else:
        print("Expected a list or a str as fl input")
        return

    if len(fl)==0:
        print("Input file list is empty")
        exit()

    fl.sort()

    # if pointsrcs:
    #     fl = range(nimages)

    for ii, fn in enumerate(fl[:]):

        if fdirout is None:
            fnout = fn.strip('.png')+'-conv.npy'
        else:
            fnout = fdirout + fn.split('/')[-1][:-4] + 'x%d.png' % rebin

        if os.path.isfile(fnout):
            print(fnout)
            print("File exists, skipping.")
            continue

        if ii%10==0:
            print("Finished %d/%d" % (ii, len(fl)))

        if pointsrcs:
            print(fn)
            Nx, Ny = NSIDE, NSIDE
            data = np.zeros([Nx,Ny])
            nsrc = np.random.poisson(int(src_density*(Nx*Ny*PIXEL_SIZE**2/60.**2)))
            fdirgalparams = fdirout[:-6]+'/galparams/'
            if not os.path.isdir(fdirgalparams):
                os.system('mkdir %s' % fdirgalparams)
            fnblobout = fdirgalparams + fn.split('/')[-1].strip('.png')+'GalParams.txt'
            SimObj = simulation.SimRadioGal(nx=Nx, ny=Ny)
            data = SimObj.sim_sky(distort_gal=False, fnblobout=fnblobout)

            if len(data.shape)==2:
                data = data[..., None]
            norm = True
        elif sky:
            data = np.load('SKA-fun-model.npy')
            data = data[800:800+4*118, 800:800+4*124]
            mm=np.where(data==data.max())[0]
            data[data<0] = 0
            data /= (data.max()/255.0/12.)
            data[data>255] = 255
            data = data.astype(np.uint8)
            data = data[..., None]
        else:
            data = cv2.imread(fn)
        
        #noise_arr = np.random.normal(0, 0.005*data.max(), data.shape)
        #data += noise_arr.astype(data.dtype)

        if distort_psf:
            for aa in [1]:
                kernel_ = kernel[..., None]*np.ones([1,1,3])
#                alphad = np.random.uniform(0,5)
                alphad = np.random.uniform(0,20)
                if plotit:
                    plt.subplot(131)
                    plt.imshow(kernel,vmax=0.1,)
                kernel_ = elastic_transform(kernel_, alpha=alphad,
                                           sigma=3, alpha_affine=0)
                if plotit:
                    plt.subplot(132)
                    plt.imshow(kernel_[..., 0], vmax=0.1)
                    plt.subplot(133)
                    plt.imshow(kernel-kernel_[..., 0],vmax=0.1, vmin=-0.1)
                    plt.colorbar()
                    plt.show()

                kernel_ = kernel_[..., 0]
                fdiroutPSF = fdirout[:-6]+'/psf/'
                fnout1=fdirout+'./test%0.2f.png'%aa
                fnout2=fdirout+'./test%0.2fx4.png'%aa
                np.save(fdiroutPSF+fn.split('/')[-1][:-4] + '-%0.2f-.npy'%alphad, kernel_)
        else:
            kernel_ = kernel

        dataLR, data_noise = convolvehr(data, kernel_, plotit=plotit, 
                                        rebin=rebin, norm=norm, nbit=nbit, 
                                        noise=True)

        data = normalize_data(data, nbit=nbit)
        dataLR = normalize_data(dataLR, nbit=nbit)

        if nbit==8:
            if save_img:
                cv2.imwrite(fnout, dataLR.astype(np.uint8))
            else:
                np.save(fnout[:-4], dataLR)
        elif nbit==16:
            if save_img:
                cv2.imwrite(fnout, dataLR.astype(np.uint16))
            else:
                np.save(fnout[:-4], dataLR)

        if pointsrcs or sky:
            fnoutHR = fdirout + fn.split('/')[-1][:-4] + '.png'
            fnoutHRnoise = fdirout + fn.split('/')[-1][:-4] + 'noise.png'

            if nbit==8:
                if save_img:
                    cv2.imwrite(fnoutHR, data.astype(np.uint8))
                else:
                    np.save(fnoutHR, data)
            elif nbit==16:
                if save_img:
                    cv2.imwrite(fnoutHR, data.astype(np.uint16))
                    cv2.imwrite(fnoutHRnoise, data_noise.astype(np.uint16))
                else:
                    np.save(fnoutHR, data)

        del dataLR, data, data_noise
 
if __name__=='__main__':
    # Example usage:
    # DIV2K: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 32 -o ./images/PSF-pointsrc-4x/test/ -p -r 4
    # Point sources: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 32 -o ./images/PSF-pointsrc-4x/test/ -p -r 4 -x
    # SKA sky image: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 64 -o ./images/PSF-pointsrc-4x/test/ -p --sky -r 2

    parser = optparse.OptionParser(prog="hr2lr.py",
                   version="",
                   usage="%prog input_dir kernel  [OPTIONS]",
                   description="Take high resolution images, convolve them, \
                   and save output.")

    parser.add_option('-d', dest='fdirin', type='str',
                      help="input directory")
    parser.add_option('-k', '--kernel', dest='kernel', type='str',
                      help="", default='Gaussian')
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=64)
    parser.add_option('-o', '--fdout', dest='fdout', type='str',
                      help="output directory", default=None)
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")
    parser.add_option('-x', '--pointsrcs', dest='pointsrcs', action="store_true",
                      help="only do point sources")
    parser.add_option('--sky', dest='sky', action="store_true",
                      help="use SKA mid image as input")
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits for image", default=16)
    parser.add_option('-n', '--nchan', dest='nchan', type=int,
                      help="number of frequency channels for image", default=1)
    parser.add_option('--scp', dest='scp', action="store_true",
                      help="scp data to cms-imaging")
    parser.add_option('--distort_psf', dest='distort_psf', action="store_true",
                      help="perturb PSF for each image generated")

    options, args = parser.parse_args()
    if options.kernel.endswith('npy'):
        kernel = np.load(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
    elif options.kernel in ('Gaussian', 'gaussian'):
        kernel1D = signal.gaussian(8, std=1).reshape(8, 1)
        kernel = np.outer(kernel1D, kernel1D)
    elif options.kernel.endswith('fits'):
        from skimage import transform
        kernel, header, pixel_scale_psf, num_pix = readfits(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
        pixel_scale_psf *= 3600
        if abs((1-pixel_scale_psf/PIXEL_SIZE)) > 0.025:
            print("Stretching PSF by %0.3f to match map" % (pixel_scale_psf/PIXEL_SIZE))
            kernel = transform.rescale(kernel, pixel_scale_psf/PIXEL_SIZE)

    fdirinTRAIN = options.fdirin+'/DIV2K_train_HR/'
    fdirinVALID = options.fdirin+'/DIV2K_valid_HR/'
    fdiroutTRAIN = options.fdout+'/train/'
    fdiroutVALID = options.fdout+'/valid/'
    fdiroutPSF = options.fdout+'/psf/'

    if not os.path.isdir(fdiroutTRAIN):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutTRAIN)

    if not os.path.isdir(fdiroutVALID):
        print("Making output validation directory")
        os.system('mkdir -p %s' % fdiroutVALID)

    if not os.path.isdir(fdiroutPSF):
        print("Making output PSF directory")
        os.system('mkdir -p %s' % fdiroutPSF)

    create_LR_image(fdirinTRAIN, kernel, fdirout=fdiroutTRAIN, 
            plotit=options.plotit, pointsrcs=options.pointsrcs, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit, 
            distort_psf=options.distort_psf, nchan=options.nchan)   
    create_LR_image(fdirinVALID, kernel, fdirout=fdiroutVALID, 
            plotit=options.plotit, pointsrcs=options.pointsrcs, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit,
            distort_psf=options.distort_psf, nchan=options.nchan)

    if not options.distort_psf:
        np.save('%s/psf.npy' % fdiroutPSF, kernel)

    if options.scp:
        fdirTRAINCMS = '/scratch/imaging/projects/dsa2000-sr/super-resolution/images-temp/train/'
        fdirVALIDCMS = '/scratch/imaging/projects/dsa2000-sr/super-resolution/images-temp/valid/'
        os.system('scp %s cms-imaging:%s' % (fdiroutTRAIN+'/*.png', fdirTRAINCMS))
        os.system('scp %s cms-imaging:%s' % (fdiroutVALID+'/*.png', fdirVALIDCMS))






