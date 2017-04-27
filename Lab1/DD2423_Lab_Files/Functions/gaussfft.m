function out_pic = gaussfft(pic, t)

%To build the mesh grid we need to know the size of the image
[x_size y_size] = size(pic);
%what we do now is generate 2 vectors x and y, that both goes from -Size/2
%to (size/2 -1)
[x_mesh y_mesh] = meshgrid(-x_size/2 : (x_size/2)-1, -y_size/2 : (y_size/2)-1);
% now we apply the formula for the Gaussian function in the spatial domain
gauss_funct = (1/(2*pi*t))*exp(-(x_mesh.*x_mesh + y_mesh.*y_mesh)/(2*t));
%We transform it to the domain space
gauss_funct_hat = fft2(gauss_funct);
%we transform the input pic
pic_hat = fft2(pic);
%the output pic is the multiplication in the fourier domain of the
%gaussian filter and the pic. We apply then the inverse transformation
% and we shift the image to adapt the coordinates.


out_pic = fftshift(ifft2(gauss_funct_hat .* pic_hat));


end


