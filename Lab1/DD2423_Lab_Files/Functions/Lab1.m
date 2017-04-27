%%we prepare matlab
clear
close all
% we use fftwave function to generate all the images
% we need: Real, imaginarty, abs, and angle
%All the images will be 128x128 pixel big

%%Question  1 
Size = 128; 
%The first Point != 0 is in the coordinates
p = 5;
q = 9;

 

%Assigned 1 to that coordinate
figure;
fftwave(p , q, Size );
%The next step is to try different points (p,q)
Values =[9,5;17,9;17,121;5,1;125,1];

for i=1:size(Values,1)
    figure;
    fftwave(Values(i,1),Values(i,2),Size);
end

%% Question 2

close all;
clear all; 
% we choose a point in the left upper side of the image so we dont need to
% transform the coordinates
u = 4;
v = 8;
Size = 128;
img = zeros(Size,Size);
re = zeros(Size,Size);
sol = zeros(Size,Size);
%Now we need to get the value in the spatial domain of every pixel
%(128*128) then we need a loop like the following
for m = 1:Size
    for n = 1:Size
            f = 2 * pi * ((m*u + n*v)/Size);
            sol(m,n) = 1/Size *( cos(f) + i*sin(f));
            re(m,n) = 1/Size * cos(f) ;
            img(m,n) = 1/Size * i*sin(f) ;
    end
end


%We show first the propagation on the x axis
figure(1)
for i=1:Size
    plot(real(sol(:,i)))
    pause(0.01)
end

%We show first then the propagation on the y axis
figure(2)
for i=1:Size
    plot(real(sol(i,:)))
    pause(0.01)
end



%% Question 5.A

% It now exceeds in the x direction

clear all
close all

Size = 128;

u = 5;
v = 9;

figure
fftwave(u,v,Size)    
figure
fftwave(u,Size+2-v,Size)  % we use Size + 2 as we want the exact same symetric point but on the right side of the image
                          % we check it as now the wave length is the same
%% Question 5.B

%It now exceeds in the y direction

clear all
close all

Size = 128;

u = 5;
v = 9;

figure
fftwave(u,v,Size)    
figure
fftwave(Size + 2 - u,v,Size)  % we use Size + 2 as we want the exact same symetric point but on the right side of the image
                          % we check it as now the wave length is the same

%% 5.2 - exceed on both

%It now exceeds in both directions

clear all
close all

Size = 128;

u = 5;
v = 9;

figure
fftwave(u,v,Size)    
figure
fftwave(Size + 2 - u,Size+2-v,Size)  % we use Size + 2 as we want the exact same symetric point but on the right side of the image
                          % we check it as now the wave length is the same
%% Question 7 - Linearity
clear all
close all

%We declare threee test images
F = [ zeros(56,128); ones(16,128); zeros(56,128)];
G = F';
H = F + 2*G;

% Display this images

figure
subplot(2,2,1)
showgrey(F)
title('F');
subplot(2,2,2)
showgrey(G);
title('G');
subplot(2,2,3)
showgrey(H);
title('H');

%We compute now the Fourier transformation of the images

Fhat = fft2(F);
Ghat = fft2(G);      
Hhat = fft2(H);
          
%We now show both the original image and the transformation          1 + 

figure
subplot(3,2,1)
showgrey(F)
title('F');
subplot(3,2,2)
showgrey(log((1+abs(Fhat))));
title('Fhat');
subplot(3,2,3)
showgrey(G);
title('G');
subplot(3,2,4)
showgrey(log(1+abs(Ghat)));
title('Ghat');
subplot(3,2,5)
showgrey(H);
title('H');
subplot(3,2,6)
showgrey(log(1+abs(Hhat)));
title('Hhat');

%And the extra one, compared to the original H one.
figure
subplot(2,2,1)
showgrey(H)
title('H');
subplot(2,2,2)
showgrey(log(1+abs(Hhat)));
title('Hhat');
subplot(2,2,3)
showgrey(H)
title('H');
subplot(2,2,4)
showgrey(log(1+abs(fftshift(Hhat))));
title('Hhat_shift');


%we reorder the quadrants and center the origin


figure
subplot(3,2,1)
showgrey(F)
title('F');
subplot(3,2,2)
showgrey(log(1+abs(fftshift(Fhat))));
title('Fhat shifted');
subplot(3,2,3)
showgrey(G);
title('G');
subplot(3,2,4)
showgrey(log(1+abs(fftshift(Ghat))));
title('Ghat shifted');
subplot(3,2,5)
showgrey(H);
title('H');
subplot(3,2,6)
showgrey(log(1+abs(fftshift(Hhat))));
title('Hhat shifted');
%% Question 8
% show the difference between using the function log or not
clear all
close all

%We declare threee test images
F = [ zeros(56,128); ones(16,128); zeros(56,128)];
G = F';
H = F + 2*G;

%We compute now the Fourier transformation of the images

Fhat = fft2(F);
Ghat = fft2(G);      
Hhat = fft2(H);

figure
subplot(3,2,1)
showgrey(log(1+abs(fftshift(Fhat))));
title('Fhat shifted');
subplot(3,2,2)
showgrey(1+abs(fftshift(Fhat)));
title('Fhat shifted no log');
subplot(3,2,3)
showgrey(log(1+abs(fftshift(Ghat))));
title('Ghat shifted');
subplot(3,2,4)
showgrey(1+abs(fftshift(Ghat)));
title('Ghat shifted no log');
subplot(3,2,5)
showgrey(log(1+abs(fftshift(Hhat))));
title('Hhat shifted');
subplot(3,2,6)
showgrey(1+abs(fftshift(Hhat)));
title('Hhat shifted no log');

%we show now how log changes the function
figure
for i=1:128
    x(i) = i;
end
y = 1+abs(fftshift(Fhat));
m = log(1+abs(fftshift(Fhat)));
plot(x,y, 'r')
figure
plot(x,m, 'b')
%% Question 9
clear all
close all

%We declare threee test images
F = [ zeros(56,128); ones(16,128); zeros(56,128)];
G = F';
H = F + 2*G;

%We compute now the Fourier transformation of the images

Fhat = fft2(F);
Ghat = fft2(G);      
Hhat = fft2(H);

%show all the three transformation together
figure
subplot(2,2,1)
showgrey(log(1+abs(fftshift(Fhat))));
title('Fhat shifted');
subplot(2,2,2)
showgrey(log(1+abs(fftshift(Ghat))));
title('Ghat shifted');
subplot(2,2,3)
showgrey(log(1+abs(fftshift(Hhat))));
title('Hhat shifted');
%we will show now the linearity 
figure
subplot(1,2,1)
showfs(Hhat);
title('Hhat');
subplot(1,2,2)
showfs(Fhat+2*Ghat);
title('Fhat + 2*Ghat');

%% Question 10

clear all
close all
Size = 128;
%We declare threee test images
F = [ zeros(56,128); ones(16,128); zeros(56,128)];
G = F';

subplot(1,3,1)
showgrey( F .* G);
title('F .* G');
subplot(1,3,2)
showfs (fft2(F .* G));
title('F(F .*G )');
subplot(1,3,3)
%Is divided by Size^2 as fft function uses Size^2 instead 
% of Size. 
showfs (fft2(F)*fft2(G)/(Size^2));
title('F(F)*F(G)');

%% Question 11

clear all 
close all

F = [zeros(60,128); ones(8,128); zeros(60,128);] .* [zeros(128,48) ones(128,32) zeros(128,48)];
G = [ zeros(56,128); ones(16,128); zeros(56,128)];
Fhat = fft2(F);
Ghat = fft2(G.*G');

subplot(2,2,1)
showgrey( F);
title('F');
subplot(2,2,2)
showfs (Fhat);
title('Fhat');
subplot(2,2,3)
showgrey(G.*G');
title('Previous excercise spatial domain');
subplot(2,2,4)
showfs (Ghat);
title('Previous excercise Fourier domain');

%% Question 12

clear all 
close all

alpha = 30;
F = [zeros(60,128); ones(8,128); zeros(60,128);] .* [zeros(128,48) ones(128,32) zeros(128,48)];
G = rot(F,alpha);

Fhat = fft2(F);
Ghat = fft2(G);

subplot(3,2,1)
showgrey( F);
title('F');
subplot(3,2,2)
showfs (Fhat);
title('Fhat');
subplot(3,2,3)
showgrey(G);
title('G');
subplot(3,2,4)
showfs (Ghat);
title('Ghat');

Hhat = rot( fftshift(Ghat), -alpha);

subplot(3,2,5)
showgrey(log(1 + abs(Hhat)));
title('Ghat reverted');

%% Question 12, show angles

clear all 
close all

F = [zeros(60,128); ones(8,128); zeros(60,128);] .* [zeros(128,48) ones(128,32) zeros(128,48)];

Fhat = fft2(F);

alpha = [ 0 30 45 60 90 ];
[dumb angles] = size(alpha);
cont = 1; 

for i=1:angles
    G = rot(F,alpha(i));
    Ghat = fft2(G);
    
    subplot(angles,3,cont)
    showgrey(G);
    title(sprintf('G => %d', alpha(i)));
    cont = cont + 1; 
    subplot(angles,3,cont)
    showfs (Ghat);
    title(sprintf('Ghat => %d', alpha(i)));
    cont = cont + 1; 
    Hhat = rot( fftshift(Ghat), -alpha(i));

    subplot(angles,3,cont)
    showgrey(log(1 + abs(Hhat)));
    title(sprintf('Ghat unrotated => %d', -alpha(i)));
    cont = cont + 1; 
end 

%% Question 13

%  phonecalc128, few128, nallo128 and study the 
%  results on screen (for very small values of a ? 10?10)
%  An image can be loaded like this:
%  img = phonecalc128.

close all
clear all

num_images = 3;

subplot(num_images,num_images,1)
showgrey(phonecalc128)
title('phonecalc128 normal');
subplot(num_images,num_images,2)
showgrey(pow2image(phonecalc128,10e-10))
title('phonecalc128 pow');
subplot(num_images,num_images,3)
showgrey(randphaseimage(phonecalc128))
title('phonecalc128 rand phase');
subplot(num_images,num_images,4)
showgrey(few128)
title('few128 normal');
subplot(num_images,num_images,5)
showgrey(pow2image(few128,10e-10))
title('few128 pow');
subplot(num_images,num_images,6)
showgrey(randphaseimage(few128))
title('few128 rand phase');
subplot(num_images,num_images,7)
showgrey(nallo128 )
title('nallo128 normal');
subplot(num_images,num_images,8)
showgrey(pow2image(nallo128 ,10e-10))
title('nallo128 pow');
subplot(num_images,num_images,9)
showgrey(randphaseimage(nallo128 ))
title('nallo128 rand phase');

%% Question 14 Show the impulse response and variance 
%for the above mentioned t-values.
%What are the variances of your discretized Gaussian kernel 
%for t = 0.1, 0.3, 1.0, 10.0 and 100.0?

close all
clear all

values = [ 0.1 0.3 1.0 10.0 100.0 ];
[dumb max_t] = size(values);
Cov_psf = zeros(2,2,max_t);
Var = zeros(max_t);
Ideal_var = zeros(max_t);
Ideal_Cov = zeros(2,2,max_t);
figure
for t = 1:max_t
    psf = gaussfft(deltafcn(128, 128), values(t));
    Cov_psf(:,:,t) = variance(psf);
    Var(t) = Cov_psf(1,1,t);
    Ideal_Cov(:,:,t) = values(t)*[1 0 ; 0 1];
    Ideal_var(t) = Ideal_Cov(1,1,t);
    subplot(max_t,1,t)
    showgrey(psf)
	title(sprintf('Impulse response - t = %f - Variance = %f - Ideal = %f', values(t), Var(t), Ideal_var(t)));
end

gaussfft_show(values, deltafcn(128, 128),1 );
 
%% Question 16 - Question 16: Convolve a couple of images 
%with Gaussian functions of different variances 
%(like t = 1.0, 4.0, 16.0, 64.0 and 256.0) and present
%your results. What effects can you observe?

close all
clear all

values = [0.000000001 1.0 4.0 16.0 64.0 256.0 ];

phone = phonecalc128;
few = few128;
nallo = nallo128;

num_images = 3; 

[dumb max_t] = size(values);
Cov_psf = zeros(2,2,max_t);
Var = zeros(max_t);
Ideal_var = zeros(max_t);
Ideal_Cov = zeros(2,2,max_t);
figure

%phone pic
for t = 1:max_t
    psf = gaussfft(phone, values(t));
    Cov_psf(:,:,t) = variance(psf);
    Var(t) = Cov_psf(1,1,t);
    Ideal_Cov(:,:,t) = values(t)*[1 0 ; 0 1];
    Ideal_var(t) = Ideal_Cov(1,1,t);
    subplot(num_images,max_t,t)
    showgrey(psf)
    title(sprintf('Impulse response - t = %f ', values(t)));
end

%few pic
for t = 1:max_t
    psf = gaussfft(few, values(t));
    Cov_psf(:,:,t) = variance(psf);
    Var(t) = Cov_psf(1,1,t);
    Ideal_Cov(:,:,t) = values(t)*[1 0 ; 0 1];
    Ideal_var(t) = Ideal_Cov(1,1,t);
    subplot(num_images,max_t,t + max_t)
    showgrey(psf)
    title(sprintf('Impulse response - t = %f ', values(t)));
end

%nallo pic
for t = 1:max_t
    psf = gaussfft(phone, values(t));
    Cov_psf(:,:,t) = variance(psf);
    Var(t) = Cov_psf(1,1,t);
    Ideal_Cov(:,:,t) = values(t)*[1 0 ; 0 1];
    Ideal_var(t) = Ideal_Cov(1,1,t);
    subplot(num_images,max_t,t + 2*max_t)
    showgrey(psf)
    title(sprintf('Impulse response - t = %f ', values(t)));
end

gaussfft_show(values, phone,0);

%% Question 17 - What are the positive and negative 
%effects for each type of filter? Describe
%what you observe and name the effects that you recognize. 
%How do the results depend on the filter parameters? 
%Illustrate with Matlab figure(s).
 close all
 clear all
 
office = office256;
add = gaussnoise(office, 16);
sap = sapnoise(office, 0.1, 255);

figure
subplot(1,3,1);
showgrey(office); 
title('Original');
subplot(1,3,2);
showgrey(add); 
title('Gaussian noise');
subplot(1,3,3);
showgrey(sap);
title('SAP noise');

%we use now the different filters for the different images
t_param = [ 1.0 4.0 16.0 64.0 256.0 ];
[dumb max_t] = size(t_param);
%Do is the cutoff parameter of the ideal filter, it has to be lower than 1
Do_param = [0.3 0.1 0.05 0.025 0.01 ];
[dumb max_d] = size(Do_param);
%Median paraMETER
Med_param = [ 1 2 4 8 16 ];
[dumb max_m] = size(Med_param);
%For the add pic - gauss noise
figure
%Gaussian filter
subplot(3,max_t+1,1)
showgrey(add)
title('Original Gaussian noise image');
for t = 1:max_t
    subplot(3,max_t+1,t+1)
    filtered_image = gaussfft(add,t_param(t));
    showgrey(filtered_image)
    title(sprintf('Gaussian filter - t = %f ', t_param(t)));
end
%Ideal filter
subplot(3,max_d+1,max_t+2)
showgrey(add)
title('Original Gaussian noise image');
for d = 1:max_d
    subplot(3,max_d+1,d+2+max_d)
    filtered_image = ideal(add,Do_param(d));
    showgrey(filtered_image)
    title(sprintf('Ideal filter - Do = %f ', Do_param(d)));
end
%Ideal filter
subplot(3,max_m+1,2*(max_t)+3)
showgrey(add)
title('Original Gaussian noise image');
for m = 1:max_m
    subplot(3,max_m+1,m+3+2*max_m)
    filtered_image = medfilt(add,Med_param(m));
    showgrey(filtered_image)
    title(sprintf('Median filter - Med. Param = %f ', Med_param(m)));
end

%For the SAP pic - salt and pepper noise
figure
%Gaussian filter
subplot(3,max_t+1,1)
showgrey(sap)
title('Original SAP noise image');
for t = 1:max_t
    subplot(3,max_t+1,1+t)
    filtered_image = gaussfft(sap,t_param(t));
    showgrey(filtered_image)
    title(sprintf('Gaussian filter - t = %f ', t_param(t)));
end
%Ideal filter
subplot(3,max_d+1,max_t+2)
showgrey(sap)
title('Original SAP noise image');
for d = 1:max_d
    subplot(3,max_d+1,d+2+max_d)
    filtered_image = ideal(sap,Do_param(d));
    showgrey(filtered_image)
    title(sprintf('Ideal filter - Do = %f ', Do_param(d)));
end
%Ideal filter
subplot(3,max_m+1,2*(max_t)+3)
showgrey(sap)
title('Original SAP noise image');
for m = 1:max_m
    subplot(3,max_m+1,m+3+2*max_m)
    filtered_image = medfilt(sap,Med_param(m));
    showgrey(filtered_image)
    title(sprintf('Median filter - Med. Param = %f ', Med_param(m)));
end

%% Question 19 

close all
clear all

% as it is said to be done in the lab notes - Gaussian smoothing
img = phonecalc256;
smoothimgGauss = img;
smoothimgIdeal = img;
N=5;
t = 1; 
d = 0.1;
figure
for i=1:N
    if i>1 % generate subsampled versions
        img = rawsubsample(img);
        smoothimgGauss = gaussfft(smoothimgGauss,t);
        smoothimgGauss = rawsubsample(smoothimgGauss);
        smoothimgIdeal = gaussfft(smoothimgIdeal,t);
        smoothimgIdeal = rawsubsample(smoothimgIdeal);
    end
    subplot(3, N, i)
    showgrey(img)
    title(sprintf('subsampled original i = %i', i));
    subplot(3, N, i+N)
    showgrey(smoothimgGauss)
    title(sprintf('subsampled Gaussian smoothed i = %i', i));
    subplot(3, N, i + 2*N)
    showgrey(img)
    title(sprintf('subsampled ideal i = %i', i));
end

