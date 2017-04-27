function [linepar accumulator] = houghline(curves, magnitude, nrho, ntheta, threshold, nlines, verbose, iterations_smooth, smooth_t, subploted, h, accum, hough)
% linepar is a list of (?, ?) parameters for each line segment,
% acc is the accumulator matrix of the Hough transform,
% curves are the polygons from which the transform is to be computed,
% magnitude is an image with one intensity value per pixel (in exercise 6.2 you will here give the gradient magnitude as an argument),
% nrho is the number of accumulators in the ? direction,
% nthetais the number of accumulators in the ? direction,
% threshold is a threshold for the magnitude,
% nlines is the number of lines to be extracted,
% verbose denotes the degree of extra information and figures that will be shown.
%SMOOTHING PARAMTERS

ACCM = accum;



% Check if input appear to be valid
if (nrho < 1 || ntheta < 1 || nlines < 1 || size(curves,1) < 2 )
    display(' ERROR ');
    return;
end

% Allocate accumulator space
accumulator = zeros(nrho,ntheta);

% Define a coordinate system in the accumulator space
%Theta is given in pixel and rho in radians, then 
% the max value of rho will be the distance from one corner to another
%this distance is, 
dist = sqrt((size(magnitude,1)-1)^2 + (size(magnitude,2)-1)^2);
%We substract one as the origin is in (1,1)
rho_space = linspace(-dist,dist,nrho);
%Using the same reasoning, rho can only go from -90 to 90 degres
degre = pi/2;
%We compute the coordinates
theta_space=linspace(-degre,degre,ntheta);
%We compute deltaRho and deltaTheta, which is a unitary increment in the
%coordintes rho and theta

delta_rho = norm(rho_space(2) - rho_space(1));
delta_theta = norm(theta_space(2) - theta_space(1));

%Curves is a vector of points. The first y value determines the number of
%points. 
insize = size(curves, 2);
trypointer = 1;

numcurves = 0;
image=zeros(size(magnitude));

while trypointer <= insize,
      %numer of points of the first curve  
      polylength = curves(2, trypointer);
      %Increase of the loop counter so now we work with the actual points.
      %The firs parameter was only the length
      trypointer = trypointer + 1;
      % For each point on each curve
      for polyidx = 1:polylength
        x = curves(2, trypointer);
        y = curves(1, trypointer);
        %we are using this variables as indexes so we need integreger values
        roundx = round(x);
        roundy = round(y);
        %We save the values greater than the threshold
        thres_mg = magnitude > threshold;
        %If it passed the magnitude then
        if thres_mg(roundx, roundy)
            %We need this to output the accumulator
            image(roundx,roundy) = 1;
            % Loop over all posible theta values
            for i=1:ntheta
               % Compute posible rho for each theta value and each point
               rho = x * cos(theta_space(i)) + y * sin(theta_space(i));
               % Compute index values in the accumulator space
               %We save a one in the position of the bin where rho falls
               %We check rho_space (-rho to rho) and assign a 1 to the
               %bigger value smaller than rho.
               rho_bin = find(rho_space < rho, 1, 'last');
               % Update the accumulator. We add one just to increment the
               % value of the accumulator in the specific coordinates that
               % we have obtained
               if ACCM <= 0
                   accumulator(rho_bin,i) = accumulator(rho_bin,i) + 1;
               else
                   accumulator(rho_bin,i) = accumulator(rho_bin,i) + h*magnitude(roundx,roundy);
               end
               
            end
        end
        %To check every point
        trypointer = trypointer + 1;
      end
end

%If we need to smooth the accumulator histogram as the lab notes suggested
if iterations_smooth ~= 0
    accumulator = binsepsmoothiter(accumulator,smooth_t,iterations_smooth);
end

%Output the curves and the accumulator
if ( verbose >= 2)
    figure;
    subplot(1,3,1);
    overlaycurves(magnitude,curves);
    axis([1 size(magnitude,1) 1 size(magnitude,2)])
    title('Edges')
    subplot(1,3,2);
    showgrey(accumulator);
    title('Accumulator')
end

%We locate the local maxima from the accumulator and its indexes
[pos, value] = locmax8(accumulator);
%We save the indexes of the sorted value vector
[~,indexvector] = sort(value);
%We save the number of local maxima values
nmaxima = size(value, 1);
% Compute a line for each one of the strongest responses in the accumulator
for idx = 1:nlines
    %we save the indexes for each maximum value of rho and theta in
    %decreasing order of magnitude. 
    rhoidxacc = pos(indexvector(nmaxima - idx + 1), 1);
    thetaidxacc = pos(indexvector(nmaxima - idx + 1), 2);
    %we get the values for this strongest responses in an 'out' vector for
    %theta and another for rho
    rho_out = rho_space(rhoidxacc);
    theta_out = theta_space(thetaidxacc);
    
    %We have to choose thre points now for every line
    %Initial values, chosen by us. We choose the midle point
    x0 = size(magnitude,2)/2;
    %From the equation
    y0 = (rho_out-x0*cos(theta_out))/sin(theta_out);
    %Big one, we need lines bigger than the image so the output is correct.
    %As we set the x0 in the middle, we only need to add/subtract the same
    %ammount of pixels
    dx = size(magnitude,2)/2;
    %Trigonometry
    dy =-cos(theta_out)/sin(theta_out)*dx;
       
    %Lecture notes
    linepar(1, 4*(idx-1) + 1) = 0; % level, not significant
    linepar(2, 4*(idx-1) + 1) = 3; % number of points in the curve
    linepar(2, 4*(idx-1) + 2) = x0-dx;
    linepar(1, 4*(idx-1) + 2) = y0-dy;
    linepar(2, 4*(idx-1) + 3) = x0;
    linepar(1, 4*(idx-1) + 3) = y0;
    linepar(2, 4*(idx-1) + 4) = x0+dx;
    linepar(1, 4*(idx-1) + 4) = y0+dy;
    %Out put this points in the accumulator
    if ( verbose >= 2)
        subplot(1,3,2);
        hold on;
        plot(thetaidxacc,rhoidxacc,'rx');
        hold off;
    end
end

% Overlay these curves on the gradient magnitude image
if ( verbose >= 1)
    if subploted <= 0
        if ( verbose >= 2 )
            subplot(1,3,3);
        else
            figure;
        end
        overlaycurves(image,linepar);
        axis([1 size(magnitude,1) 1 size(magnitude,2)])
        title('Hough lines')
    end
    if subploted > 0 && hough > 0
        showgrey(accumulator)
    end
end


end