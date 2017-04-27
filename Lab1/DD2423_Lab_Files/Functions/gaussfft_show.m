function gaussfft_show(values_v,pic, Spatial)

[dumb max_t] = size(values_v);
figure
max = max_t/2;
max = ceil(max_t/2);
%phone pic
for t = 1:max_t
   
   %To build the mesh grid we need to know the size of the image
    [x_size y_size] = size(pic);
    %what we do now is generate 2 vectors x and y, that both goes from -Size/2
    %to (size/2 -1)
    [x_mesh y_mesh] = meshgrid(-x_size/2 : (x_size/2)-1, -y_size/2 : (y_size/2)-1);
    % now we apply the formula for the Gaussian function in the spatial domain
    gauss_funct = (1/(2*pi*t))*exp(-(x_mesh.*x_mesh + y_mesh.*y_mesh)/(2*values_v(t)));
    %We transform it to the domain space
    gauss_funct_hat = fft2(gauss_funct);
    %we transform the input pic
    pic_hat = fft2(pic);
    %output of the distribution, used in Q15 and Q16
    if Spatial == 0
        %Fourier domain
        subplot(max,max,t)
        surf(fftshift(gauss_funct_hat));
        title(sprintf('Fourier domain - t = %f ', values_v(t)));
    else
    %Spatial domain
        subplot(max,max,t)
        surf(gauss_funct);
        title(sprintf('Spatial domain - t = %f ', values_v(t)));
    end
end

end
