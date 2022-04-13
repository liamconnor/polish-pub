import sys

import numpy as np
import tensorflow as tf
from optparse import OptionParser

from data import RadioSky
from model.wdsr import wdsr_b
from train import WdsrTrainer

def main(images_dir, caches_dir, fnoutweights, ntrain=800, nvalid=100,
         scale=4, nchan=1, nbit=16, num_res_blocks=32, batchsize=4,
         train_steps=10000):

    train_loader = RadioSky(scale=scale,  # 2, 3, 4 or 9
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='train',
                     images_dir=images_dir,
                     caches_dir=caches_dir,
                     nchan=nchan,
                     ntrain=ntrain,
                     nvalid=nvalid)       # Training dataset are images 001 - 800

    # Create a tf.data.Dataset         
    train_ds = train_loader.dataset(batch_size=batchsize,  # batch size as described in the EDSR and WDSR papers
                                random_transform=True, # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

    valid_loader = RadioSky(scale=scale,             # 2, 3, 4 or 8
                     downgrade='bicubic', # 'bicubic', 'unknown', 'mild' or 'difficult' 
                     subset='valid',
                     images_dir=images_dir,
                     caches_dir=caches_dir,
                     ntrain=ntrain,
                     nvalid=nvalid)      # Validation dataset are images 801 - 900
                     
    # Create a tf.data.Dataset          
    valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False, # use DIV2K images in original size 
                                repeat_count=1)         # 1 epoch



    trainer = WdsrTrainer(model=wdsr_b(scale=scale, num_res_blocks=num_res_blocks, nchan=nchan), 
                      checkpoint_dir=f'.ckpt/%s'%fnoutweights.strip('.h5'))

    # Train WDSR B model for train_steps steps and evaluate model
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
    parser.add_option("-c", "--cachdir", dest="caches_dir", default=None,
                      help="directory with training/validation image data")
    parser.add_option("-f", "--fnout", dest="fnout_model", type=str, default='model.h5',
                      help="directory with training/validation image data")
    parser.add_option("-x", "--scale", dest="scale", type=int, default=4,
                      help="upsample factor")
    parser.add_option("--nchan", dest="nchan", type=int, default=1,
                      help="number of frequency channels in images")
    parser.add_option("--num_res_blocks", dest="num_res_blocks", type=int, default=32,
                      help="number of residual blocks in neural network")
    parser.add_option("--nbit", dest="nbit", type=int, default=16,
                      help="number of bits in image data")
    parser.add_option("--train_steps", dest="train_steps", type=int, default=10000,
                      help="number of training steps")
    parser.add_option('--ntrain', dest='ntrain', type=int,
                      help="number of training images", default=800)
    parser.add_option('--nvalid', dest='nvalid', type=int,
                      help="number of validation images", default=100)

    options, args = parser.parse_args()
    images_dir = args[0]

    if options.caches_dir is None:
        if images_dir[-1]=='/':
            caches_dir = images_dir[:1] + '-cache'
        else:
            caches_dir = images_dir + '-cache'
    else:
        caches_dir = options.caches_dir

    main(images_dir, caches_dir, options.fnout_model, 
         ntrain=options.ntrain,
         nvalid=options.nvalid,
         scale=options.scale, 
         nchan=options.nchan, 
         nbit=options.nbit, 
         num_res_blocks=options.num_res_blocks, 
         train_steps=options.train_steps)







