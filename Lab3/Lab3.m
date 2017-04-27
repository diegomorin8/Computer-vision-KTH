%% QUICK

%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('tiger1.jpg');
K = 8;               % number of clusters used
L = 5;              % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range

I1 = kmeans_example(I,K,L,seed,scale_factor,image_sigma,0);
imshow(I1)

%% Q1 - Output compare between initialization methods - QUICK

%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('orange.jpg');
K = 8;               % number of clusters used
L = [1 3 5 8 14 20];         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence

L_max= size(L,2);

for i = 1:L_max
    subplot(2,L_max + 1,i)
    imshow( kmeans_example(I,K,L(i),seed,scale_factor,image_sigma,0, check_ite));
    title(sprintf(' 0 - 255 rand L = %i',L(i)));
    subplot(2,L_max + 1,i + L_max + 1)
    imshow( kmeans_example(I,K,L(i),seed,scale_factor,image_sigma,2,  check_ite));
    title(sprintf(' Bandwidth - average L = %i',L(i)));
end   

subplot(2,L_max + 1, 7)
imshow(I);
title('Original');
subplot(2,L_max + 1, 14)
imshow(I);
title('Original');
    
%% Q1 - plot the distance differences - super slow

%Is too long, just for the report


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
K = 8;               % number of clusters used
L = 45;         % number of iterations
seed = 30;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 1; % 1- check convergence, 0- dont check convergence
I_max =  size(I,1);
graph = 1; % to output the distances difference

for i = 1:I_max
    disp(sprintf( I(i,:)))
    disp(sprintf('0 - 256 rand '))
    subplot(2,I_max,i)
    kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,0, check_ite, graph);
    title(sprintf('0 - 256 rand - %s', I(i,:)));
    disp(sprintf('Band width around average'))
    subplot(2,I_max,i + I_max)
    kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,2,  check_ite, graph);
    title(sprintf('Bandwidth - average - %s', I(i,:)));
end   


%% Q1 and Q2 -  Check the convergence for the different initialization - SUUUUPERSLOW

%Is too long, just for the report


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
K = 8;               % number of clusters used
L = 100;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 1; % 1- check convergence, 0- dont check convergence
I_max =  size(I,1);


for i = 1:I_max
    disp(sprintf( I(i,:)))
    disp(sprintf('0 - 256 rand '))
    subplot(2,I_max,i)
    imshow( kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,0, check_ite));
    title(sprintf('0 - 256 rand ', I(i,:)));
    disp(sprintf('Band width around average'))
    subplot(2,I_max,i + I_max)
    imshow( kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,2,  check_ite));
    title(sprintf('Band width around average', I(i,:)));
end   


%% Q1 and Q2 -  Check the convergence for the different initialization - SUUUUPERSLOW

%Is too long, just for the report


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
K = 8;               % number of clusters used
L = 100;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 1; % 1- check convergence, 0- dont check convergence
I_max =  size(I,1);


for i = 1:I_max
    disp(sprintf( I(i,:)))
    disp(sprintf('0 - 256 rand '))
    subplot(2,I_max,i)
    imshow( kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,0, check_ite));
    title('0 - 256 rand ');
    disp(sprintf('0 - max__value rand '))
    subplot(2,I_max,i + I_max)
    imshow( kmeans_example(imread(I(i,:)),K,L,seed,scale_factor,image_sigma,2,  check_ite));
    title('Band width around average' );
end   


%% Q2

I = imread('tiger1.jpg');
K = 8;               % number of clusters used
L = 5;             % number of iterations
seed = 30;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
alpha = 8.0;                 % maximum edge cost
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
area = [ 80, 110, 570, 300 ]; % image region to train foreground with
%Let see the difference between the methods
I1 = graphcut_example(I,K,seed,scale_factor,image_sigma,alpha,area,Mean);

%% Q2 - convergence changes


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('orange.jpg');
K = 8;               % number of clusters used
L = [8 10 12 14 16 18 20];         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence

L_max= size(L,2);

for i = 1:L_max
    subplot(2,(L_max +1)/2,i)
    imshow( kmeans_example(I,K,L(i),seed,scale_factor,image_sigma,0, check_ite));
    title(sprintf(' 0 - 255 rand L = %i',L(i)));
end   

subplot(2,(L_max +1)/2, 8)
imshow(I);
title('Original');

%% Q3 - K effect


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('orange.jpg');
K = [2 3 6 7 8 9 10];               % number of clusters used
L = 10;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence
out_lines = 1; %output the boundaries of each cluster
K_max= size(K,2);

for i = 1:K_max
    subplot(2,(K_max +1)/2,i)
    imshow( kmeans_example(I,K(i),L,seed,scale_factor,image_sigma,0, check_ite,0, out_lines));
    title(sprintf(' 0 - 255 rand K = %i',K(i)));
end   

subplot(2,(K_max +1)/2, 8)
imshow(I);
title('Original');

%% Q4 - Same parameters in tiger 1 and 2


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all


K = 10;               % number of clusters used
L = 10;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence
out_lines = 0; %output the boundaries of each cluster

I = imread('tiger1.jpg');
subplot(1,2,1)
imshow( kmeans_example(I,K,L,seed,scale_factor,image_sigma,0, check_ite,0, out_lines));
title(sprintf(' 0 - 255 rand K = %i',K));
I = imread('tiger2.jpg');
subplot(1,2,2)
imshow( kmeans_example(I,K,L,seed,scale_factor,image_sigma,0, check_ite,0, out_lines));
title(sprintf(' 0 - 255 rand K = %i',K));

%% Q4 - K effect - tiger1 - QUICK


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('tiger1.jpg');
K = [5 6 7];               % number of clusters used
L = 10;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 1.0;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence
out_lines = 0; %output the boundaries of each cluster

K_max= size(K,2);

for i = 1:K_max
    subplot(2,(K_max +1)/2,i)
    imshow( kmeans_example(I,K(i),L,seed,scale_factor,image_sigma,0, check_ite,0, out_lines));
    title(sprintf(' 0 - 255 rand K = %i',K(i)));
end   

subplot(2,(K_max +1)/2, 4)
imshow(I);
title('Original');


%% Q4 - K effect - tiger2 - sigma QUICK


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('tiger2.jpg');
K = 10;               % number of clusters used
L = 10;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = [ 0.2 0.5 1.2 1.6];   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence
out_lines = 0; %output the boundaries of each cluster

S_max= size(image_sigma,2);

for i = 1:S_max
    subplot(2,(S_max)/2,i)
    imshow( kmeans_example(I,K,L,seed,scale_factor,image_sigma(i),0, check_ite,0, out_lines));
    title(sprintf(' 0 - 255 rand Sigma = %f',image_sigma(i)));
end   


%% Q4 - K effect - tiger2 - sigma low - QUICK


%For the first question we need to experiment how many iterations we need
%to converge depending of the random initialisation we use to see if there
%are any diferences. 
%We can also erase the parts of the output image that doesn't change
clear all
close all

I = imread('tiger2.jpg');
K = [7 10 14 20];               % number of clusters used
L = 10;         % number of iterations
seed = 14;           % seed used for random initialization
scale_factor = 1.0;  % image downscale factor
image_sigma = 0.50;   % image preblurring scale
Mean = 2; % 2 - band width, 1- max rand, 0 - whole range
check_ite = 0; % 1- check convergence, 0- dont check convergence
out_lines = 0; %output the boundaries of each cluster

K_max= size(K,2);

for i = 1:K_max
    subplot(2,(K_max)/2,i)
    imshow( kmeans_example(I,K(i),L,seed,scale_factor,image_sigma,0, check_ite,0, out_lines));
    title(sprintf(' 0 - 255 rand K = %i',K(i)));
end   

%% TOO SLOW - Q5 color bandwidth

clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
scale_factor = 0.6;       % image downscale factor
spatial_bandwidth = [5.0 10.0 20.0 30.0 40.0 50.0];  % spatial bandwidth
colour_bandwidth = [2.0 5.0 10.0 20.0 30.0 40.0];   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  



sp_max = size(spatial_bandwidth,2);
cl_max = size(colour_bandwidth,2);

figure
for i = 1:cl_max
    subplot(3,cl_max,i)
    [~,Iseg] = mean_shift_example( imread(I(1,:)), spatial_bandwidth(4), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i', I(1,:),spatial_bandwidth(i)))
    subplot(3,cl_max,i + cl_max)
    [~,Iseg] = mean_shift_example( imread(I(2,:)), spatial_bandwidth(4), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i', I(2,:),spatial_bandwidth(i)))
    subplot(3,cl_max,i + 2*cl_max)
    [~,Iseg] = mean_shift_example( imread(I(3,:)), spatial_bandwidth(4), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i ', I(3,:),spatial_bandwidth(i)))
end  


%% TOO SLOW - Q5 Spatial bandwidth

clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
scale_factor = 0.6;       % image downscale factor
spatial_bandwidth = [5.0 10.0 20.0 30.0 40.0 50.0];  % spatial bandwidth
colour_bandwidth = [2.0 5.0 10.0 20.0 30.0 40.0];   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  



sp_max = size(spatial_bandwidth,2);
cl_max = size(colour_bandwidth,2);

figure
for i = 1:sp_max
    subplot(3,sp_max,i)
    [~,Iseg] = mean_shift_example( imread(I(1,:)), spatial_bandwidth(i), colour_bandwidth(4), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i', I(1,:),spatial_bandwidth(i)))
    subplot(3,sp_max,i + sp_max)
    [~,Iseg] = mean_shift_example( imread(I(2,:)), spatial_bandwidth(i), colour_bandwidth(4), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i', I(2,:),spatial_bandwidth(i)))
    subplot(3,sp_max,i + 2*sp_max)
    [~,Iseg] = mean_shift_example( imread(I(3,:)), spatial_bandwidth(i), colour_bandwidth(4), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i', I(3,:),spatial_bandwidth(i)))
end  

%% Quick enough - Q5 mean shift segmentation- Parameters setting

clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
scale_factor = 0.6;       % image downscale factor
% spatial_bandwidth = [5.0 5.0 10.0 20.0 7.0 5];  % spatial bandwidth
% colour_bandwidth = [ 40.0 2.0 20.0 10.0 20.0 40.0];   % colour bandwidth
 spatial_bandwidth = [7.0 5.0 3.0 5.0];  % spatial bandwidth
 colour_bandwidth = [20.0 40.0 55 50];   % colour bandwidth


num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  



sp_max = size(spatial_bandwidth,2);
cl_max = size(colour_bandwidth,2);

for i = 1:sp_max
    subplot(2,sp_max/2,i)
    [~,Iseg] = mean_shift_example( imread(I(1,:)), spatial_bandwidth(i), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i Col = %i', I(1,:),spatial_bandwidth(i),colour_bandwidth(i)));
end  

%% FASTER ONLY TIGER1 - Q5 mean shift segmentation- lets play with the parameters - Spatial

clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
scale_factor = 0.6;       % image downscale factor
% spatial_bandwidth = [5.0 5.0 10.0 20.0 7.0 5];  % spatial bandwidth
% colour_bandwidth = [ 40.0 2.0 20.0 10.0 20.0 40.0];   % colour bandwidth
 spatial_bandwidth = [ 5 10 10 10 ];  % spatial bandwidth
 colour_bandwidth = [15 20.0 10 15];   % colour bandwidth


num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  



sp_max = size(spatial_bandwidth,2);
cl_max = size(colour_bandwidth,2);

for i = 1:sp_max
    subplot(2,sp_max/2,i)
    [~,Iseg] = mean_shift_example( imread(I(2,:)), spatial_bandwidth(i), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i Col = %i', I(2,:),spatial_bandwidth(i),colour_bandwidth(i)));
end  


%% FASTER ONLY TIGER2 - Q5 mean shift segmentation- lets play with the parameters - Spatial

clear all
close all

I = [ 'orange.jpg' ;'tiger1.jpg' ;'tiger2.jpg'];
scale_factor = 0.6;       % image downscale factor
%  spatial_bandwidth = [5.0 5.0 10.0 20.0 7.0 5];  % spatial bandwidth
%  colour_bandwidth = [ 40.0 2.0 20.0 10.0 20.0 40.0];   % colour bandwidth
 spatial_bandwidth = [7.0 5.0 3.0 7.0];  % spatial bandwidth
 colour_bandwidth = [20.0 40.0 55 40];   % colour bandwidth


num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale  



sp_max = size(spatial_bandwidth,2);
cl_max = size(colour_bandwidth,2);

for i = 1:sp_max
    subplot(2,sp_max/2,i)
    [~,Iseg] = mean_shift_example( imread(I(3,:)), spatial_bandwidth(i), colour_bandwidth(i), num_iterations, scale_factor, image_sigma);
    imshow(Iseg);
    title(sprintf('%s for Spatial = %i Col = %i', I(3,:),spatial_bandwidth(i),colour_bandwidth(i)));
end  

%%

imshow(graphcut_example)
