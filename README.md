# Image Segmentation and Crowd Density Estimator
I did upsampling and convolution.In all architectures,the fully convolutional ideais followed, the structure resembles the cannonical classification CNN, as convolution, ReLU, and max pooling are repeatedly applied to the input image and feature maps.Each architecture consists of a down-sampling path, followed by an up-sampling path.In the second half of the architecture, spatial resolution is recovered by performing up-sampling, convolution, eventually mapping the intermediate feature representation back to the original resolution.In the U-net version, low-level feature representations are fused during upsampling, aiming to compensate the information loss due to max pooling.

reference:
https://towardsdatascience.com/objects-counting-by-estimating-a-density-map-with-convolutional-neural-networks-c01086f3b3ec for U-net architecture
https://anandology.com/blog/using-iterators-and-generators/
