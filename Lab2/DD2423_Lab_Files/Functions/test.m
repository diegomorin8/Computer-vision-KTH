function pixels = test(type,input,shape)
    
    if (nargin < 3)
        shape = 'valid';
    end
    %%We have to make sure that all the masks have the same sizes in all
    %%the directions. We use the central masks as it was said in the lab
    %%notes. 
    
    %%5x5 matrix (Lab notes). Done to make sure that all masks work
    %%consistently in all directions
    
    %First order
    xmask = zeros(5,5);
    ymask = zeros(5,5);
    xmask(3,2:4) = [-1/2 0 1/2];
    ymask(2:4,3) = [1/2 0 -1/2]';
    
    %Second order
    xxmask = zeros(5,5);
    yymask = zeros(5,5);
    xxmask(3,2:4) = [1 -2 1];
    yymask(2:4,3) = [1 -2 1]';
    
    %third order
    convxy = conv2(xmask,ymask,shape);
    convyyy = conv2(ymask,yymask,shape);
    convxxx = conv2(xmask,xxmask,shape);
    convxxy = conv2(xxmask,ymask,shape);
    convxyy = conv2(xmask,yymask,shape);
    
    %Apply the filters
    switch type
        case 'deltax'
            pixels = filter2(xmask, input, shape);
        case 'deltaxx'
            pixels = filter2(xxmask, input, shape);
        case 'deltaxxx'
            pixels = filter2(convxxx, input, shape);
        case 'deltay'
            pixels = filter2(ymask, input, shape);
        case 'deltayy'
            pixels = filter2(yymask, input, shape);
        case 'deltayyy'
            pixels = filter2(convyyy, input, shape);
        case 'deltaxy'
            pixels = filter2(convxy, input, shape);
        case 'deltaxxy'
            pixels = filter2(convxxy, input, shape);
        case 'deltaxyy'
            pixels = filter2(convxyy, input, shape);
    end   
end
