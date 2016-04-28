
function poolingForward{T}(x::Array{T,4}, y; window=2, padding=0, stride=window, mode=0)
    stride = isa(stride, Integer) ? (stride, stride) : stride
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    if any(map(x->x>0,padding))
        x0=x
        w,h,c,n = size(x0)
        x=zeros(eltype(x0),w+2padding[1],h+2padding[2],c,n)
        x[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x0
    end
    fill!(y,0)
    @assert (mode==0)
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    @inbounds for n in 1:Nx, c in 1:C, jy in 1:Hy, iy in 1:Wy
        # iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Wx ? Wx : i+window[1]-1
        y[iy,jy,c,n] = maximum(x[i:wx_end,j:hx_end,c,n])
    end
    return y
end

# mode == 0 maxpooling
function poolingBackward{T}(y::Array{T,4}, dy::Array{T,4}, x::Array{T,4}, dx::Array{T,4}; window=2, padding=0, stride=window, mode=0)
    stride = isa(stride, Integer) ? (stride, stride) : stride
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    fill!(dx,0)
    @assert mode==0
    # x: (W,H,C,N)
    if any(map(x->x>0,padding))
        x0=x
        w,h,c,n = size(x0)
        x=zeros(eltype(x0),w+2padding[1],h+2padding[2],c,n)
        x[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x0
    end
    dx1 = zeros(x)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    # @inbounds for n in 1:Nx, c in 1:C, j in 1:stride[2]:Hx, i in 1:stride[1]:Wx
    @inbounds for n in 1:Nx, c in 1:C, jy in 1:Hy, iy in 1:Wy
        #= iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Hx ? Hx : i+window[1]-1 =#
        i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Wx ? Wx : i+window[1]-1
        a = x[i:wx_end,j:hx_end,c,n]
        di,dj = ind2sub(a,indmax(a))
        # dx[i+di-1-padding[1],j+dj-1-padding[2],c,n] += dy[iy,jy,c,n]
        dx1[i+di-1,j+dj-1,c,n] += dy[iy,jy,c,n]
        any(map(x->x>0,padding)) && (dx[:,:,c,n] = dx1[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2],c,n])
    end
    # @show dx1
    # dx = dx1[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2],:,:]
    return dx
end

function getPoolingNdForwardOutputDim{T}(x::Array{T,4}; window=2, padding=0, stride=1, mode=0)
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    stride = isa(stride, Integer) ? (stride,stride) : stride
    @assert reduce(&, [w>p for (p,w) in zip(padding,window)])
    dims = [size(x)...]
    for i=1:length(dims)-2
        # dims[i] = 1 + ceil((dims[i] + 2*padding[i] - window[i]) / stride[i])
        dims[i] = length(1:stride[i]:dims[i]+padding[i])
    end
    tuple(dims...)
end
