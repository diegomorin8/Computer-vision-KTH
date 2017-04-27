function Inew = graphcut_example(I,K,seed,scale_factor,sigma,alpha,area,Mean);
if nargin < 1
    I = imread('tiger1.jpg');
    seed = 30;
    scale_factor = 0.5;          % image downscale factor
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    K = 16;                      % number of mixture components
    alpha = 8.0;                 % maximum edge cost
    sigma = 10.0;                % edge cost decay factor
    Mean = 2;
elseif nargin < 2
    scale_factor = 0.5;          % image downscale factor
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    K = 16;                      % number of mixture components
    alpha = 8.0;                 % maximum edge cost
    sigma = 10.0;                % edge cost decay factor
    seed = 30;
    Mean = 2;
elseif nargin < 3
    scale_factor = 0.5;          % image downscale factor
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    alpha = 8.0;                 % maximum edge cost
    sigma = 10.0;                % edge cost decay factor
    seed = 30;
    Mean = 2;
elseif nargin < 4
    scale_factor = 0.5;          % image downscale factor
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    alpha = 8.0;                 % maximum edge cost
    sigma = 10.0;                % edge cost decay factor
    Mean = 2;
elseif nargin < 5
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    alpha = 8.0;                 % maximum edge cost
    sigma = 10.0;                % edge cost decay factor
    Mean = 2;
elseif nargin < 6
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    Mean = 2;
    alpha = 8.0;                 % maximum edge cost
elseif nargin < 7
    area = [ 80, 110, 570, 300 ]; % image region to train foreground with
    Mean = 2;
elseif nargin < 8
    Mean = 2;
end


I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alpha, sigma, seed, Mean);

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
imwrite(Inew,'result/graphcut1.png')
imwrite(I,'result/graphcut2.png')
imwrite(prior,'result/graphcut3.png')
subplot(2,2,1); imshow(Inew);
subplot(2,2,2); imshow(I);
subplot(2,2,3); imshow(prior);
