using Base.Test
using CUDArt
using CUDNN
using CNN
@show CUDNN_VERSION


srand(7)
@show padding=2
@show stride=2
vrange = 0:2
x = Array{Float32}(rand(vrange,8,8,1,1)); tx = CudaArray(x); @show x
w = Array{Float32}(rand(vrange,3,3,1,1)); tw = CudaArray(w); @show w
@show ydims = cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)
@assert cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride) == getConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
y = zeros(Float32,ydims); ty = CudaArray(y);
cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty); @show y
y2 = zeros(Float32,size(y));
convolutionForward(x,w,y2; padding=padding, stride=stride); @show y2
@test_approx_eq y y2

@show x
dy = Array{Float32}(rand(vrange,size(y))); tdy = CudaArray(dy); @show dy
dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
cudnnConvolutionBackwardFilter(tx,tdy,tdw; padding=padding, stride=stride); dw = to_host(tdw); @show dw
dw2 = zeros(Float32, size(w));
convolutionBackwardFilter(x,dy,dw2; padding=padding, stride=stride); @show dw2
@test_approx_eq dw dw2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
cudnnConvolutionBackwardData(tw, tdy, tdx; padding=padding, stride=stride); dx = to_host(tdx); @show dx
dx2 = zeros(Float32, size(x));
convolutionBackwardData(w, dy, dx2; padding=padding, stride=stride); @show dx2
@test_approx_eq dx dx2
