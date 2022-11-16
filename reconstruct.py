import os
import sys

import optparse
import matplotlib.pylab as plt

from model import resolve_single
from utils import load_image, plot_sample
from model.wdsr import wdsr_b

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

def reconstruct(fn_img, fn_model, scale, fnhr=None,
                nbit=16, nchan=1):
    if fn_img.endswith('npy'):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith('png'):
      try:
          datalr = load_image(fn_img)
      except:
          return 

    if fnhr is not None:
        if fnhr.endswith('npy'):
            datalr = np.load(fnhr)[:, :]
        elif fnhr.endswith('png'):
          try:
              datahr = load_image(fnhr)
          except:
              return 
    else:
        datahr = None

    model = wdsr_b(scale=scale, num_res_blocks=32, nchan=nchan)
    model.load_weights(fn_model)

    if len(datalr.shape)==2:
      datalr = datalr[:,:,None]
      
    print(datalr.shape)

    datasr = resolve_single(model, datalr, nbit=nbit)
    datasr = datasr.numpy()
    return datalr, datasr, datahr

def plot_reconstruction(datalr, datasr, datahr=None, vm=1, 
                        nsub=2, cmap='afmhot'):
    """ Plot the dirty image, POLISH reconstruction, 
    and (optionally) the high resolution true sky image
    """
    vminlr=0
    vmaxlr=22500
    vminsr=0
    vmaxsr=22500
    vminhr=0
    vmaxhr=22500

    if nsub==2:
        fig = plt.figure(figsize=(10,6))
    if nsub==3:
        fig = plt.figure(figsize=(13,6))
    ax1 = plt.subplot(1,nsub,1)
    plt.title('Dirty map', color='C1', fontsize=17)
    plt.axis('off')
    plt.imshow(datalr[...,0], cmap=cmap, vmax=vmaxlr, vmin=vminlr, 
               aspect='auto', extent=[0,1,0,1])
    plt.setp(ax1.spines.values(), color='C1')
    
    ax2 = plt.subplot(1,nsub,2, sharex=ax1, sharey=ax1)
    plt.title('POLISH reconstruction', c='C2', fontsize=17)
    plt.imshow(datasr[...,0], cmap=cmap, vmax=vmaxsr, vmin=vminsr, 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')

    if nsub==3:
        ax3 = plt.subplot(1,nsub,3, sharex=ax1, sharey=ax1)
        plt.title('True sky', c='k', fontsize=17)
        plt.imshow(datahr, cmap=cmap, vmax=vmaxsr, vmin=vminsr, 
                  aspect='auto', extent=[0,1,0,1])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = optparse.OptionParser(prog="hr2lr.py",
                                   version="",
                                   usage="%prog image weights.h5  [OPTIONS]",
                                   description="Take high resolution images, deconvolve them, \
                                   and save output.")

    parser.add_option('-f', dest='fnhr', 
                      help="high-res file name", default=None)
    parser.add_option('-x', dest='scale', 
                      help="spatial rebin factor", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits in image", default=16)
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")

    options, args = parser.parse_args()
    fn_img, fn_model = args

    datalr, datasr, datahr = reconstruct(fn_img, fn_model, options.scale, 
                                 fnhr=options.fnhr, nbit=options.nbit)

    if datahr is not None:
        nsub = 3 
    else:
        nsub = 2

    if options.plotit:
        plot_reconstruction(datalr, datasr, datahr=datahr, vm=1, nsub=nsub)







