using Base.Test
using CUDArt
using CUDNN
using CNN
@show CUDNN_VERSION

srand(7)

function ptest()
    w,h = rand(10:20,2)
    window = rand(6:9,2)
    padding = rand(0:5,2)
    stride = rand(1:9,2)
    x = Array{Float32}(rand(0:2,w,h,3,10)); tx = CudaArray(x);
    ydims = getPoolingNdForwardOutputDim(x, window=window, padding=padding, stride=stride, mode=0)
    @show w, h, window, padding, stride, ydims
    y = zeros(Float32, ydims); ty = CudaArray(y);
    cudnnPoolingForward(tx, ty; window=window, padding=padding, stride=stride, mode=0); y = to_host(ty);

    y2 = zeros(y)
    poolingForward(x, y2; window=window, padding=padding, stride=stride, mode=0);
    @test_approx_eq y y2

    dy = Array{Float32}(rand(0:2,size(y))); tdy = CudaArray(dy);
    dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
    cudnnPoolingBackward(ty, tdy, tx, tdx; window=window, padding=padding, stride=stride, mode=0); dx = to_host(tdx);

    dx2 = zeros(Float32, size(x));
    poolingBackward(y, dy, x, dx2; window=window, padding=padding, stride=stride, mode=0)
    @test_approx_eq dx dx2
end

for i in 1:100 ptest(); print(".") end
