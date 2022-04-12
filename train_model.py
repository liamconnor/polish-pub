import sys

import numpy as np
import tensorflow as tf
from optparse import OptionParser

from data import RadioSky
from model.wdsr import wdsr_b
from train import WdsrTrainer

# nchan = 1
# nbit = 16
# scale = 3
# batchsize = 4
# num_res_blocks = 32
# train_steps = 1000000
# ntrain = 800
# fnoutweights = 'vla-nodistort-1e6steps-1arcmin-deeper2.h5'
# #fnoutweights = 'vla-newnoise-2x-1M-headtail.h5'
# #fnoutweights = 'AJ-15x60s-4000chan-0.5arcsec-3x-1M.h5'
# #images_dir = sys.argv[1]
# #caches_dir='./caches-%s' % images_dir
# images_dir = './fullband-5may-3x/'
# images_dir = 'vla-data-gregg-3x/'
# caches_dir='caches-vla-data-gregg-3x/'

def main(images_dir, caches_dir, fnoutweights, ntrain=800,
         scale=4, nchan=1, nbit=16, num_res_blocks=32, batchsize=4,
         train_steps=10000):

    train_loader = RadioSky(scale=scale,  # 2, 3, 4 or 9
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train',
                     images_dir=images_dir,
                     caches_dir=caches_dir,
#                     nchan=nchan,
                     ntrain=ntrain)       # Training dataset are images 001 - 800

    # Create a tf.data.Dataset         
    train_ds = train_loader.dataset(batch_size=batchsize,  # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

    valid_loader = RadioSky(scale=scale,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid',
                     images_dir=images_dir,
                     caches_dir=caches_dir)      # Validation dataset are images 801 - 900
                     
    # Create a tf.data.Dataset          
    valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch



    trainer = WdsrTrainer(model=wdsr_b(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan), 
                      checkpoint_dir=f'.ckpt/vla-faker')

    # Train WDSR B model for 300,000 steps and evaluate model
    # every 1000 steps on the first 10 images of the DIV2K
    # validation set. Save a checkpoint only if evaluation
    # PSNR has improved.

    trainer.train(train_ds,
                  valid_ds.take(10),
                  steps=train_steps,
                  evaluate_every=1000, 
                  save_best_only=True)

    trainer.restore()
    # Evaluate model on full validation set.
    psnr = trainer.evaluate(valid_ds)
    print(f'PSNR = {psnr.numpy():3f}')

    # Save weights to separate location.
    trainer.model.save_weights(fnoutweights)

if __name__=='__main__':
    parser = OptionParser(prog="train_model",
                                   version="",
                                   usage="%prog fname datestr specnum [OPTIONS]",
                                   description="Visualize and classify filterbank data")
    parser.add_option("-c", "--cachdir", dest="caches_dir", type=str, default=None,
                      help="directory with training/validation image data")
    parser.add_option("-f", "--fnout", dest="fnout_model", type=str, default=None,
                      help="directory with training/validation image data")

    options, args = parser.parse_args()
    images_dir = args[0]

    main(images_dir, options.caches_dir, options.fnout_model, 
         ntrain=800,
         scale=3, nchan=1, nbit=16, 
         num_res_blocks=32, 
         train_steps=10000)
