using Base.Test
using CUDArt
using CUDNN
using CNN
@show CUDNN_VERSION

srand(7)

using CUDNN: PD
x = Array{Float32}(rand(0:2,4,5,1,1)); tx = CudaArray(x); @show x
@show size(x)
@show window, padding, stride = (3,3), (1,1), (3,2)
pd1 = PD(2, window, padding, stride, CUDNN_POOLING_MAX)
@show cudnnGetPoolingNdForwardOutputDim(pd1, tx) # CUDNN.cudnnGetPoolingNdForwardOutputDim is BUGGY!!!
@show ydims = getPoolingNdForwardOutputDim(x, window=window, padding=padding, stride=stride, mode=0)
# @show ydims = cudnnGetPoolingNdForwardOutputDim(pd1, tx)
y = zeros(Float32, ydims); ty = CudaArray(y);
cudnnPoolingForward(tx, ty; window=window, padding=padding, stride=stride, mode=0); y = to_host(ty); @show y

y2 = zeros(y)
poolingForward(x, y2; window=window, padding=padding, stride=stride, mode=0); @show y2
@test_approx_eq y y2

#
dy = Array{Float32}(rand(0:2,size(y))); tdy = CudaArray(dy); @show dy
dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
cudnnPoolingBackward(ty, tdy, tx, tdx; window=window, padding=padding, stride=stride, mode=0); dx = to_host(tdx); @show dx

dx2 = zeros(Float32, size(x));
poolingBackward(y, dy, x, dx2; window=window, padding=padding, stride=stride, mode=0); @show dx2
@test_approx_eq dx dx2

