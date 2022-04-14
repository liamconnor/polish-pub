# polish-pub

The POLISH radio interferometric image reconstruction algorithm. 

https://arxiv.org/abs/2111.03249

The code in this repository generates high resolution (HR) simulated images of the radio sky as well as the convolved, low resolution (LR) dirty images produced by radio interferometers. It can then train a POLISH neural network on these image pairs to make a learning-based, feed forward imaging model for super-resolved radio images. 

The super-resolution neural network and the code to train it comes from the WDSR implementation found here:

https://github.com/krasserm/super-resolution

# using POLISH

Start by constructing a training/validation dataset. This will consist of a set of image pairs (true sky / dirty image) such that POLISH can learn deconvolution. You will require a PSF for the interferometer whose data you wish to deconvolve. This will produce 800 training image pairs, 100 validation image pairs using a forward-modeled dsa-2000 PSF. The radio sky simulation assumes the approximate senstivity of DSA-2000, i.e. SEFD=2.5 Jy. By default, these images will be 2048x2048 (true sky, high res, 0.25'' pixels) and 512x512 (dirty image, low res, 1'' pixels), but these parameters can be changed with command line arguments.
```
% python make_img_pairs.py -o dsa-example -k psf/dsa-2000-fullband-psf.fits -s 512
```
The data will be stored in various subdirectories of dsa-example, which can be fed directly to the POLISH trainer. To train on these data simply run
```
% python train_model.py dsa-example -f dsa-example-model.h5 
```
