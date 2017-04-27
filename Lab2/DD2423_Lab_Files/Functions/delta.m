function operator = delta(axis,input)

x = 1; 
y = 2; 
%Own conclusion, as the developed applycation of the operator is the same
%but not multiplied by 1/2; 
simplex = [-1 0 1];
simpley = [1 0 -1]'; 

%lab notes
centralx = [-1/2 0 1/2];
centraly = [1/2 0 -1/2]'; 

%lab notes
central2x = [1 -2 1];
central2y = [1 -2 1]'; 

%http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/edges.pdf
robertx = [ 1 0; 0 -1];
roberty = [ 0 1; -1 0];

sobelx = [-1 0 1; -2 0 2; -1 0 1];
sobely = sobelx';

if axis == x
    switch input
        case 'simple'
            operator = simplex;
        case 'central'
            operator = centralx; 
        case 'central2'
            operator = central2x; 
        case 'robert'
            operator = robertx;
        case 'sobel'
            operator = sobelx;
    end
else
    switch input
        case 'simple'
            operator = simpley;
        case 'central'
            operator = centraly; 
        case 'central2'
            operator = central2y; 
        case 'robert'
            operator = roberty;
        case 'sobel'
            operator = sobely;
    end
end

end