# polish-pub

The POLISH radio interferometric image reconstruction algorithm. 

https://arxiv.org/abs/2111.03249

The code in this repository generates high resolution (HR) simulated images of the radio sky as well as the convolved, low resolution (LR) dirty images produced by radio interferometers. It can then train a POLISH neural network on these image pairs to make a learning-based, feed forward imaging model for producing super-resolved radio images. 

The super-resolution neural network and the code to train it comes from the WDSR implementation found here:

https://github.com/krasserm/super-resolution
