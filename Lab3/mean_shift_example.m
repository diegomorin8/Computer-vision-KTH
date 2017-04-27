function [I,Inew] = mean_shift_example( I, spatial_bandwidth, colour_bandwidth, num_iterations, scale_factor, image_sigma)

if nargin < 1
I = imread('orange.jpg');
scale_factor = 0.5;       % image downscale factor
spatial_bandwidth = 20.0;  % spatial bandwidth
colour_bandwidth = 10.0;   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  
elseif nargin < 2
scale_factor = 0.5;       % image downscale factor
spatial_bandwidth = 20.0;  % spatial bandwidth
colour_bandwidth = 10.0;   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale
elseif nargin < 2
scale_factor = 0.5;       % image downscale factor
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale
elseif nargin < 2
scale_factor = 0.5;       % image downscale factor
image_sigma = 1.0;        % image preblurring scale
elseif nargin < 2
image_sigma = 1.0;        % image preblurring scale
end

I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
% imwrite(Inew,'result/meanshift1.png')
% imwrite(I,'result/meanshift2.png')
% subplot(1,2,1); imshow(Inew);
% subplot(1,2,2); imshow(I);


