
function convy{T}(x0::SubArray{T,2}, w::SubArray{T,2}, padding::Array{Int,1}, stride::Array{Int,1}; xcorr=false)
    if any(padding .> 0) # this could be handled better...
        x=zeros(eltype(x0), 2*padding+collect(size(x0))...)
        x[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]] = x0
    else
        x=x0
    end
    row_extend, col_extend = floor(Int, 1 + (collect(size(x)) - collect(size(w))) ./ stride)
    widx = Int[sub2ind(size(x),i,j) for i in 1:stride[1]:size(x,1)-size(w,1)+1, j in 1:stride[2]:size(x,2)-size(w,2)+1] # linear indexes of filter positions in x

    oidx = Int[(j-1)*size(x,1)+i for i in 1:size(w,1), j in 1:size(w,2)] # linear indexes of elements in a filter window
    destidx = Int[i+(j-1) for i in widx, j in oidx]
    return reshape(x[destidx]*(xcorr ? w[:] : reverse(w[:])),row_extend,col_extend)
end

# dw = rot180(xcorr(x,dy))
function convdw{T}(x0::Array{T,2}, dy::Array{T,2}, w::Array{T,2}, padding::Array{Int,1}, stride::Array{Int,1})
    if any(padding .> 0) # this could be handled better...
        x=zeros(eltype(x0), 2*padding+collect(size(x0))...)
        x[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]] = x0
    else
        x=x0
    end
    x1l = last(collect(take(countfrom(1,stride[1]),size(dy,1))))
    x2l = last(collect(take(countfrom(1,stride[2]),size(dy,2))))
    widx = Int[sub2ind(size(x),i,j) for i in 1:size(w,1), j in 1:size(w,2)]
    oidx = Int[sub2ind(size(x),i,j) for i in 1:stride[1]:x1l, j in 1:stride[2]:x2l] # linear indexes of elements in a filter window
    destidx = Int[i+(j-1) for i in widx, j in oidx]
    return rot180(reshape(x[destidx]*dy[:],size(w)))
end

# dx = xcorr(dy, w, 'full')
function convdx{T}(dy::Array{T,2}, w::Array{T,2}, dx::Array{T,2}, padding::Array{Int,1}, stride::Array{Int,1})
    size_tdy = collect(size(dx)) + collect(size(w)) - 1 + 2padding
    tdy = zeros(T, size_tdy...)

    pad1, pad2 = map(x->x-1,size(w))
    for (i,idy) in zip(countfrom(pad1+1,stride[1]), 1:size(dy,1)), (j,jdy) in zip(countfrom(pad2+1,stride[2]), 1:size(dy,2))
        tdy[i,j] = dy[idy,jdy]
    end
    res = convy(sub(tdy,:,:), sub(w,:,:), [0,0], [1,1]; xcorr=true)
    return all(padding .== 0) ? res : res[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]] 
end


#=
function _conv2{T}(x::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    max_pad = map(x->x-1-pad,size(w))
    y = conv2(x, xcorr ? rot180(w) : w)
    return y[1+max_pad[1]:stride:end-max_pad[1], 1+max_pad[2]:stride:end-max_pad[2]]
end
=#

# mode == 0 convolution
# algorithm == 0 implicit_gemm
function convolutionForward{T}(x::Array{T,4}, w::Array{T,4}, y::Array{T,4}; padding=0, stride=1, 
                                    upscale=1, mode=0, cd=nothing,
                                    algorithm=0,
                                    workSpace=0, workSpaceSizeInBytes=0, alpha=1, beta=1)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    # x: (W,H,C,N)
    # w: (W,H,C,K) 
    # y: (W,H,K,N) 
    fill!(y,0)
    @assert (upscale==1 && mode==0 && algorithm == 0) "$((upscale,mode,algorithm))"
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw

    @inbounds for n in 1:N, k in 1:K, c in 1:Cx
        # y[:,:,k,n] += convy(x[:,:,c,n], w[:,:,c,k], padding, stride, xcorr=mode!=0) # TODO use sub function
        y[:,:,k,n] += convy(sub(x, :,:,c,n), sub(w,:,:,c,k), padding, stride, xcorr=mode!=0) # TODO use sub function
    end
    return y
end

# dw = rot180(xcorr(x,dy))
function convolutionBackwardFilter{T}(x::Array{T,4}, dy::Array{T,4}, dw::Array{T,4}; padding=0, stride=1, upscale=1, mode=0)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    fill!(dw,0)
    @assert (upscale==1&& mode==0)
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    @inbounds for c in 1:C, k in 1:K, n in 1:Ny
        dw[:,:,c,k] += convdw(x[:,:,c,n], dy[:,:,k,n], dw[:,:,c,k], padding, stride) # TODO: use sub function
    end
    return dw
end

# dx = xcorr(dy, w, 'full')
function convolutionBackwardData{T}(w::Array{T,4}, dy::Array{T,4}, dx::Array{T,4}; padding=0, stride=1, upscale=1, mode=0)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    fill!(dx,0)
    @assert (upscale==1&& mode==0)
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    @assert Ky==Kw
    @inbounds for n in 1:N, c in 1:C, k in 1:Kw
        dx[:,:,c,n] += convdx(dy[:,:,k,n], w[:,:,c,k], dx[:,:,c,n], padding, stride)
    end
    return dx
end

function getConvolutionNdForwardOutputDim{T}(x::Array{T,4}, w::Array{T,4}; padding=padding,stride=stride)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) ./ stride)
    return (Wy,Hy,K,N)
end



