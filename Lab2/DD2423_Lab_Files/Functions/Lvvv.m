function pixels = Lvvv(inpic,shape)
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
    convxy = conv2(xmask,ymask,shape);
    convyyy = conv2(ymask,yymask,shape);
    convxxx = conv2(xmask,xxmask,shape);
    convxxy = conv2(xxmask,ymask,shape);
    convxyy = conv2(xmask,yymask,shape);
    Lxy = filter2(convxy, inpic, shape);
    Lxxx = filter2(convxxx, inpic, shape);
    Lyyy = filter2(convyyy, inpic, shape);
    Lxxy = filter2(convxxy, inpic, shape);
    Lxyy = filter2(convxyy, inpic, shape);
    
   pixels = Lx.^3.*Lxxx+3*Lx.^2.*Ly.*Lxxy+3*Lx.*Ly.^2.*Lxyy+Ly.^3.*Lyyy;


     
end
