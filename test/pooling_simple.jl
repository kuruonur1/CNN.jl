using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION

import CUDNN.cudnnConvolutionForward
import CUDNN.cudnnConvolutionBackwardFilter
import CUDNN.cudnnConvolutionBackwardData
import CUDNN.cudnnGetConvolutionNdForwardOutputDim
import CUDNN.cudnnGetPoolingNdForwardOutputDim
import CUDNN.cudnnPoolingForward
import CUDNN.cudnnPoolingBackward
GPU=true
include(Pkg.dir("Knet/src/util/conv_pool_cpu.jl"))

srand(7)

using CUDNN: PD, CUDNN_POOLING_MAX, cudnnGetPoolingNdForwardOutputDim
# x = rand(Float32,18,18,3,100); tx = CudaArray(x); @show x
# x = reshape(Float32[1:25;], 5, 5, 1, 1); tx = CudaArray(x); @show x
x = Array{Float32}(rand(0:2,4,5,1,1)); tx = CudaArray(x); @show x
@show size(x)
@show psize, padding, stride = (3,3), (1,1), (3,2)
pd1 = PD(2, psize, padding, stride, CUDNN_POOLING_MAX)
@show cudnnGetPoolingNdForwardOutputDim(pd1, tx) # CUDNN.cudnnGetPoolingNdForwardOutputDim is BUGGY!!!
@show ydims = cudnnGetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
# @show ydims = cudnnGetPoolingNdForwardOutputDim(pd1, tx)
y = zeros(Float32, ydims); ty = CudaArray(y);
cudnnPoolingForward(tx, ty; window=psize, padding=padding, stride=stride, mode=0); y = to_host(ty); @show y

y2 = zeros(y)
cudnnPoolingForward(x, y2; window=psize, padding=padding, stride=stride, mode=0); @show y2
@test_approx_eq y y2

#
dy = Array{Float32}(rand(0:2,size(y))); tdy = CudaArray(dy); @show dy
dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
cudnnPoolingBackward(ty, tdy, tx, tdx; window=psize, padding=padding, stride=stride, mode=0); dx = to_host(tdx); @show dx

dx2 = zeros(Float32, size(x));
cudnnPoolingBackward(y, dy, x, dx2; window=psize, padding=padding, stride=stride, mode=0); @show dx2
@test_approx_eq dx dx2

