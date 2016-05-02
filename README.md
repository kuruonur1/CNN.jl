# CNN
CNN provides convolution/pooling forward/backward operations to train a Convolutional Neural Network (CNN) on CPU. It utilizes parallel matrix multiplications (gemm) for speed up and has full support for any stride and pooling size. The module serves as the backbone to power up CNNs and has similar interface to [CUDNN.jl](https://github.com/JuliaGPU/CUDNN.jl).

## Convolution
x: Input, w: Filter, y: Output
`convolutionForward(x, w, y; padding=(0,0), stride=(1,1))` This function computes and returns y, the convolution of x with filter (w) under the settings (padding=(p1,p2), stride=(s1,s2)). The settings default to (padding=(0,0), stride=(1,1)). 

`convolutionBackwardFilter(x, dy, dw; padding=(0,0), stride=(1,1))` Given x and dJ/dy (abbrev. dy: derivative with respect to the output), this function computes and returns dJ/dw (abbrev. dw: derivative with respect to the filter). Notice that forward and backward settings must match for consistency.

`convolutionBackwardData(w, dy, dx; padding=(0,0), stride=(1,1))` Given w and dJ/dy, this function computes and returns dJ/dx (abbrev. dx: derivative with respect to input).

## Pooling
`poolingForward(x, y; window=(2,2), padding=(0,0), stride=window)` Performs the pooling operation on x specified by (window, padding, stride).

`poolingBackward(y, dy, x, dx; window=(2,2), padding=(0,0), stride=window)` This function computes and returns dJ/dx (abbrev. dx) where x is the input and y is the output of forward pooling operation, dJ/dy (abbrev. dy) is the loss gradient.
