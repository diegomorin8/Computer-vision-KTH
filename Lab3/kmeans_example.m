function Inew = kmeans_example(I,K,L,seed,scale_factor,image_sigma,Mean, check_ite, graph, out_lines)

if nargin < 1
    I = imread('orange.jpg');
    K = 8;               % number of clusters used
    L = 10;              % number of iterations
    seed = 14;           % seed used for random initialization
    scale_factor = 1.0;  % image downscale factor
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 2
    K = 8;               % number of clusters used
    L = 10;              % number of iterations
    seed = 14;           % seed used for random initialization
    scale_factor = 1.0;  % image downscale factor
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 3
    L = 10;              % number of iterations
    seed = 14;           % seed used for random initialization
    scale_factor = 1.0;  % image downscale factor
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 4
    seed = 14;           % seed used for random initialization
    scale_factor = 1.0;  % image downscale factor
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 5
    scale_factor = 1.0;  % image downscale factor
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 6
    image_sigma = 1.0;   % image preblurring scale
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 7
    Mean = 2;
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 8
    check_ite = 0;
    graph = 0;
    out_lines = 0;
elseif nargin < 9
    graph = 0;
    out_lines = 0;
elseif nargin < 10   
    out_lines = 0;
end

%Main code
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

tic
[ segm, centers ] = kmeans_segm(I, K, L, seed, Mean, 1, check_ite, graph);
toc
%Apply the Kmeans
Inew = mean_segments(Iback, segm);
%Get the boundaries of the segmentation
I = overlay_bounds(Iback, segm);

if out_lines == 1
    Inew = overlay_bounds(Inew, segm);;
end
% figure
% subplot(1,2,1)
% imshow(I);
% title('Original');
% subplot(1,2,2)
% imshow(Inew);
% title('New');

end