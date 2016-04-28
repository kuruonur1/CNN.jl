using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION
using CNN

srand(7)

function ctest()
    xw,xh = rand(20:30,2)
    psize = rand(6:9,2)
    padding = rand(0:5,2)
    stride = rand(1:9,2)
    @show xw, xh, psize, padding, stride
    x = Array{Float32}(rand(0:2,xw,xh,3,10)); tx = CudaArray(x);
    w = Array{Float32}(rand(0:2,psize...,3,5)); tw = CudaArray(w);
    @show ydims = cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)
    @show getConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
    @assert cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride) == getConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
    y = zeros(Float32,ydims); ty = CudaArray(y);
    cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty);
    y2 = zeros(Float32,size(y));
    convolutionForward(x,w,y2; padding=padding, stride=stride);
    @test_approx_eq y y2

    dy = rand(Float32, size(y)); tdy = CudaArray(dy);
    dy = Array{Float32}(rand(0:2,size(y))); tdy = CudaArray(dy);
    dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
    cudnnConvolutionBackwardFilter(tx,tdy,tdw; padding=padding, stride=stride); dw = to_host(tdw)
    dw2 = zeros(Float32, size(w))
    convolutionBackwardFilter(x,dy,dw2; padding=padding, stride=stride)
    @test_approx_eq dw dw2

    dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
    cudnnConvolutionBackwardData(tw, tdy, tdx; padding=padding, stride=stride)
    dx2 = zeros(Float32, size(x));
    convolutionBackwardData(w, dy, dx2; padding=padding, stride=stride)
    @test_approx_eq dx dx2
end

for i in 1:100 ctest() end
:ok
