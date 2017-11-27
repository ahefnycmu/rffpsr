function [ Z ] = kr_product( X, Y )
%KR_PRODUCT Khatri-Rao Product (Column-wise Kronecker product)
    
    [mx, nx] = size(X);
    [my, ny] = size(Y);

    assert(nx == ny);
    Z = reshape(bsxfun(@times, reshape(X, 1, mx, nx), reshape(Y, my, 1, ny)), mx*my, nx);             
end



