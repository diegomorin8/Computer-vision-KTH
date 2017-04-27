function pixels = Lvv(inpic,shape)
    %we have to make sure the dimensionality of the mask matches, then we
    %do 
    xmask = zeros(5,5);
    ymask = zeros(5,5);
    xmask(3,2:4) = [-1/2 0 1/2];
    ymask(2:4,3) = [1/2 0 -1/2]';
    
    xxmask = zeros(5,5);
    yymask = zeros(5,5);
    xxmask(3,2:4) = [1 -2 1];
    yymask(2:4,3) = [1 -2 1]';
    
    Lx = filter2(xmask, inpic, shape);
    Ly = filter2(ymask, inpic, shape);
    Lxx = filter2(xxmask, inpic, shape);
    Lyy = filter2(yymask, inpic, shape);
    convxy = conv2(xmask,ymask, shape);
    Lxy = filter2(convxy, inpic, shape);

    pixels = Lx.^2.*Lxx+2*Lx.*Ly.*Lxy+Ly.^2.*Lyy;
     
end
