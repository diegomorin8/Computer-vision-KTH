%%Q1 - Difference operators

clear all
close all

%Load the image
tools = few256;
%To choose the direction of the gradient
x = 1;
y = 2; 
%Apply the operators 

%Simple
dxtoolsSimple = conv2(tools, delta(x,'simple'), 'valid');
dytoolsSimple = conv2(tools, delta(y,'simple'), 'valid');
%Central
dxtoolsCentral = conv2(tools, delta(x,'central'), 'valid');
dytoolsCentral = conv2(tools, delta(y,'central'), 'valid');
%Robert's diagonal
dxtoolsRobert = conv2(tools, delta(x,'robert'), 'valid');
dytoolsRobert = conv2(tools,delta(y,'robert'), 'valid');
%Sobel 
dxtoolsSobel = conv2(tools, delta(x,'sobel'), 'valid');
dytoolsSobel = conv2(tools, delta(y,'sobel'), 'valid');


%We compute the sizes
toolsize = size(tools); 
simplexSize = size(dxtoolsSimple);
simpleySize = size(dytoolsSimple);
centralxSize = size(dxtoolsCentral);
centralySize = size(dytoolsCentral);
robertxSize = size(dxtoolsRobert);
robertySize = size(dytoolsRobert); 
sobelxSize = size(dxtoolsSobel);
sobelySize = size(dytoolsSobel);

%Out put the results
figure
subplot(4,3,1)
showgrey(dxtoolsSimple)
title(sprintf('Simple dx - Size %i x %i', simplexSize(1),simplexSize(2)));
subplot(4,3,2)
showgrey(dytoolsSimple);
title(sprintf('Simple dy - Size %i x %i', simpleySize(1),simpleySize(2)));
subplot(4,3,3)
showgrey(tools);
title(sprintf('Original image - Size %i x %i', toolsize(1),toolsize(2)));
subplot(4,3,4)
showgrey(dxtoolsCentral)
title(sprintf('Central dx - Size %i x %i', centralxSize(1),centralxSize(2)));
subplot(4,3,5)
showgrey(dytoolsCentral);
title(sprintf('Central dy - Size %i x %i', centralySize(1),centralySize(2)));
subplot(4,3,6)
showgrey(tools);
title(sprintf('Original image - Size %i x %i', toolsize(1),toolsize(2)));
subplot(4,3,7)
showgrey(dxtoolsRobert)
title(sprintf('Robert dx - Size %i x %i', robertxSize(1),robertxSize(2)));
subplot(4,3,8)
showgrey(dytoolsRobert);
title(sprintf('Robert dy - Size %i x %i', robertySize(1),robertySize(2)));
subplot(4,3,9)
showgrey(tools);
title(sprintf('Original image - Size %i x %i', toolsize(1),toolsize(2)));
subplot(4,3,10)
showgrey(dxtoolsSobel)
title(sprintf('Sobel dx - Size %i x %i', sobelxSize(1),sobelxSize(2)));
subplot(4,3,11)
showgrey(dytoolsSobel);
title(sprintf('Sobel dy - Size %i x %i', sobelySize(1),sobelySize(2)));
subplot(4,3,12)
showgrey(tools);
title(sprintf('Original image - Size %i x %i', toolsize(1),toolsize(2)));

%Q1.2 Difference in using valid or same parameter. We choose sobel as it
%will be the one that changes more its size

dxtoolsSobelSame = conv2(tools, delta(x,'sobel'), 'same');
sobelxSizeSame = size(dxtoolsSobelSame);

figure
subplot(1,2,1)
showgrey(dxtoolsSobel)
title(sprintf('Soble dx valid - Size %i x %i', sobelxSize(1),sobelxSize(2)));
subplot(1,2,2)
showgrey(dxtoolsSobelSame);
title(sprintf('Soble dx same - Size %i x %i', sobelxSizeSame(1),sobelxSizeSame(2)));


%% Q2.1
close all
clear all

meth = 'sobel';
%Result to be the best one
t = 2; %Threshold for the Gaussian filter
%Smoothing with Gaussian filter
gauss_tools = discgaussfft(few256,t);
gauss_goth = discgaussfft(godthem256,t);

%we use same as we need the same image sizes for both gradient directions to compute the squared
%sumatory

tools_mask = Lv(gauss_tools,meth,'same');
house_mask = Lv(gauss_goth,meth,'same');

%Threshold used
thresholds = [0 100 500 1000 1500 2000 5000];
max_size = size(thresholds,2);

%Smoothed output
figure;
for i=1:max_size
    subplot(2,max_size,i);
    showgrey((tools_mask-thresholds(i))>0);
    title(sprintf('tools Th= %i',thresholds(i)))
    subplot(2,max_size,i+max_size);
    showgrey((house_mask-thresholds(i))>0);
    title(sprintf('godthem Th= %i',thresholds(i)))
end

%not smoothed output
tools_maskNS = Lv(few256,meth,'same');
house_maskNS = Lv(godthem256,meth,'same');

figure;
for i=1:max_size
    subplot(2,max_size,i);
    showgrey((tools_maskNS-thresholds(i))>0);
    title(sprintf('tools NS Th= %i',thresholds(i)))
    subplot(2,max_size,i+max_size);
    showgrey((house_maskNS-thresholds(i))>0);
    title(sprintf('godthem NS Th= %i',thresholds(i)))
end

%% 
%%Q4 - Test of the second order edge detectors
clear all;
close all;


%We create a matrix as it was said in the lab notes
[x,y] = meshgrid(-5:5, -5:5)

%We apply the filters to different functions
%Test function was implemented only for this part of the code.
a = test('deltax',x.^2);
b = test('deltaxx', x.^2);
c = test('deltaxxx', x.^2);
d = test('deltaxx', y.^2);


figure;
subplot(2,2,1);
showgrey(a,128);
title('\delta_{x}(x^2) = 2x^1')
subplot(2,2,2);
%Used boundaries of the function showgrey to show the difference in the
%output
%if the out put where not !n our output would be black
showgrey(b,128,-5,5);
title('\delta_{xx}(x^2) = !n')
subplot(2,2,3);
showgrey(c,128);
title('\delta_{xxx}(x^2) = 0')
subplot(2,2,4);
showgrey(d,128);
title('\delta_{xx}(y^2) = 0')


%% Q4 Q5 Q6 - Experiments

clear all;
close all;

house = godthem256;
scale = [0.0001 1.0 4.0 16.0 64.0 128.0];
[dumb max_size] = size(scale);

for i=1:max_size
    subplot(2,max_size/2,i);
    contour(Lvv(discgaussfft(house, scale(i) ), 'same'), [0 0]);
    axis('image')
    axis('ij')
    title(sprintf('scale = %i',scale(i)))
end

tools = few256;
scale = [0.0001 1.0 4.0 16.0 64.0 128.0];
[dumb max_size] = size(scale);
figure
for i=1:max_size
    subplot(2,max_size/2,i);
    showgrey(Lvvv(discgaussfft(tools, scale(i) ), 'same') < 0);
    title(sprintf('scale = %i',scale(i)))
end


%% Q7
clear all
close all

%Load the image
tools = few256;
house = godthem256;

scale = [0.0001 1.0 2.0 6.0 10.0 12.0];
max_size = size(scale,2);

th = 80; 

figure;
%Out put image
for i=1:max_size
    subplot(2,max_size,i);
  	curves = njetedge(tools,scale(i),th,'same');
    overlaycurves(tools, curves);
    title(sprintf('Tools Scale = %i',scale(i)))
    subplot(2,max_size,i+max_size);
    curves = njetedge(house,scale(i),th,'same');
    overlaycurves(house, curves);
    title(sprintf('Hools Scale = %i',scale(i)))
end


%% Q8 Hough lines

clear all
close all

triangle = triangle128;
hough_test = houghtest256;
%Load the image
tools = few256;
house = godthem256;

%houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose,iterations_smooth, smooth_t)

[linepar,acc]=houghedgeline(triangle,3,10,400,250,3,2,0);


%% Hough lines - Output

clear all
close all

triangle = triangle128;
hough_test = houghtest256;
%Load the image
tools = few256;
house = godthem256;

%houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose,iterations_smooth, smooth_t)

[linepar,acc]=houghedgeline(triangle,3,10,400,250,3,2,0);
[linepar,acc]=houghedgeline(hough_test,3,10,300,190,7,2,0);
[linepar,acc]=houghedgeline(tools,2,70,200,180,10,2,0);
[linepar,acc]=houghedgeline(house,10,2,500,420,20,1,0);
%% Q9 Thetas an rhos

%Clean everything
clear all
close all

numberRho = [150 300 500 600 800 1000 1200 1500 ];
numberTheta =[110 230 420 420 500 600 800 1000 ];
 
[sdumb max_size] = size(numberRho);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
%Out put for diffrent number of thetas and rhos. 
figure
for i=1:max_size
    subplot(2,max_size/2,i);
    tic  
  	[linepar,acc] = houghedgeline(house,10,2,numberRho(i),numberTheta(i),20,1,0,subploted);  
end

%% Q9 Thetas

%Clean everything
clear all
close all

%Now we only increase the value of theta
numberRho =  [120 1500] ;
numberTheta =[120 1500];
 
max_size = size(numberTheta,2);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
%Out put for diffrent number of thetas and rhos. 
figure

subplot(2,max_size,1);
tic  
[linepar,acc] = houghedgeline(house,10,2,numberRho(1),numberTheta(1),20,1,0,subploted);  
subplot(2,max_size,2);
tic  
[linepar,acc] = houghedgeline(house,10,2,numberRho(2),numberTheta(1),20,1,0,subploted);
subplot(2,max_size,3);
tic  
[linepar,acc] = houghedgeline(house,10,2,numberRho(1),numberTheta(1),20,1,0,subploted);  
subplot(2,max_size,4);
tic  
[linepar,acc] = houghedgeline(house,10,2,numberRho(1),numberTheta(2),20,1,0,subploted); 


%% Q10 - Compare h

clear all
close all
% 
numberRho = [ 500 600 800 1000 ];
numberTheta =[ 420 420 500 600 ];
h =  [0.001 0.1 0.5 0.8 1 2 4 8];
[max_size_h] = size(h,2);
[max_size] = size(numberTheta,2);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
%Lets check if the proportional parameter h makes the output change
figure
for i=1:max_size_h
    subplot(2,max_size_h/2,i);
    tic  
  	[linepar,acc] = houghedgeline(house,10,2,600,420,20,1,0,subploted,h(i),1);  
end

%% Q10 - compare magnitude versus regular update


clear all
close all

numberRho = [ 500 600 800 1000 ];
numberTheta =[ 420 420 500 600 ];

h =  [0.001 0.1 0.5 0.8 1 2 4 8];
[max_size_h] = size(h,2);
[max_size] = size(numberTheta,2);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
%Lets check if the proportional parameter h makes the output change
figure
for i=1:max_size*2
    subplot(2,max_size,i);
    tic
    if i <= max_size
        [linepar,acc] = houghedgeline(house,10,2,numberRho(i),numberTheta(i),20,1,0,subploted,0,0);
    else
        [linepar,acc] = houghedgeline(house,10,2,numberRho(i-max_size),numberTheta(i-max_size),20,1,0,subploted,1,1);
    end
end

%% Q10 - H comparison show accumulator

clear all
close all

numberRho = [ 500 600 800 1000 ];
numberTheta =[ 420 420 500 600 ];
h =  [0.001 1000];
[max_size_h] = size(h,2);
[max_size] = size(numberTheta,2);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
figure
for i=1:max_size_h
    subplot(2,max_size_h/2,i);
    tic  
  	[linepar,acc] = houghedgeline(house,10,2,600,420,20,1,0,subploted,h(i),1,1);  
end

%% Q10 - magnitude vs unitary accumulator

clear all
close all

numberRho = [ 500 600 ];
numberTheta =[ 420 420];
[max_size] = size(numberTheta,2);

house = godthem256;

subploted = 1; 
%function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold,
%nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
for i=1:max_size*2
    subplot(2,max_size,i);
    tic
    if i <= max_size
        [linepar,acc] = houghedgeline(house,10,2,numberRho(i),numberTheta(i),20,1,0,subploted,0,0,1);
    else
        [linepar,acc] = houghedgeline(house,10,2,numberRho(i-max_size),numberTheta(i-max_size),20,1,0,subploted,1,1,1);
    end
end

%% Q10 - magnitude vs unitary accumulator

clear all
close all

house = godthem256;
Lv(house,'central','same',1); 
title('Gradient')
