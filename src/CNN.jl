module CNN
include("conv.jl"); export convolutionForward, convolutionBackwardFilter, convolutionBackwardData, getConvolutionNdForwardOutputDim
include("pool.jl"); export poolingForward, poolingBackward, getPoolingNdForwardOutputDim

# package code goes here

end # module
