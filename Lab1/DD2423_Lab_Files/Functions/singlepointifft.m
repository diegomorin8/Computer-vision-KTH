function img = singlepointifft(x,y,size)
    img = zeros(size,size);
    % loop over all pixels in the image and compute the value to be
    % assigned to that pixel.
    for k=1:size
        for j=1:size
            % assignment is based on the discrete inverse fourier transform
            % f(m,n)=1/N^2 *
            % \Sigma_m=0^{N-1}\Sigma_n=0^{N-1}\delta(x_0+x,y_0+y)\cdot e^(2\pi i(\frac{mx+ny}{N}))
            freq = 2 * pi * ((k*x + j*y)/size);
            img(k,j) = 1/size^2 * cos(freq) + i*sin(freq);
        end
    end
end