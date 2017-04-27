function [linepar acc] = houghedgeline(pic, scale, gradmagnthreshold, nrho, ntheta, nlines, verbose, iterations_smooth,subploted,h,accum, hough)
% linepar is a list of (?, ?) parameters for each line segment,
% acc is the accumulator matrix of the Hough transform,
%pic is the grey-level image,
% scale is the scale at which edges are detected,
% gradmagnthreshold is the threshold of the gradient magnitude.
% curves are the polygons from which the transform is to be computed,
% nrho is the number of accumulators in the ? direction,
% nthetais the number of accumulators in the ? direction,
% threshold is a threshold for the magnitude,
% nlines is the number of lines to be extracted,
% verbose denotes the degree of extra information and figures that will be shown.
smooth_t = 0.5;

if nargin < 3
    nrho = size(magnitude,2);
    ntheta = 180; 
    nlines = 10; 
    grandmagnthreshold = 0; 
    verbose = 2; 
    iterations_smooth = 2;
    smooth_t = 0.5;
    subploted = 0;
    h = 0;
    accum = 0;
    hough = 0;
end
if nargin < 8
    iterations_smooth = 2;
    smooth_t = 0.5;
    subploted = 0;
    h = 0;
    accum = 0;
    hough = 0;
end
if nargin < 9
    accum = 0;
    subploted = 0;
    h = 0;
    hough = 0;
end
if nargin < 10
    accum = 0;
    h = 0;
    hough = 0;
end
if nargin < 11
    accum = 0;
    hough = 0;
end
if nargin < 12
    hough = 0;
end
% performs an edge detection step and then applies a Hough transform to the result
    %We use the function njetedge to compute the curves
    curves = njetedge(pic,scale,gradmagnthreshold,'same');
    %We get the magnitude of the gradient
    Magnitude = Lv(pic,'central','same'); 
    [xmax ymax] = size(Magnitude);
    [linepar,acc] = houghline(curves,Magnitude,nrho,ntheta,gradmagnthreshold,nlines,verbose,iterations_smooth, smooth_t,subploted,h,accum, hough);
    if verbose >= 1
        if subploted > 0
            if hough <= 0
                overlaycurves(pic, linepar);
            end
            if h <= 0 
                title(sprintf('Tools N_rho = %i Ntheta = %i Time = %.2f s Acc = %i ',nrho,ntheta,toc,accum))  
                axis([1 xmax 1 ymax])
            end
            if h > 0 
                title(sprintf('Tools N_rho = %i Ntheta = %i h = %.2f Acc = %i ',nrho,ntheta,h,accum))  
                axis([1 xmax 1 ymax])
            end
        else
            figure;
            overlaycurves(pic, linepar);
            axis([1 xmax 1 ymax])
            title(sprintf('scale= %i th= &i ',num2str(scale),num2str(gradmagnthreshold)));
        end
            
    end
end