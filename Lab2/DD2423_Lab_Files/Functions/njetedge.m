function out_curves = njetedge(inpic, scale, threshold, shape)
% ccomputes the edges of an image at arbitrary scale and returns these lists of edge points. If
% the parameter threshold is given, this value shall be used as threshold for the gradient magnitude
% computed at the same scale. Otherwise, no thresholding shall be used and all edge segments will be
% returned.

%We calculate the first derivate
First_derivate = Lv(discgaussfft(inpic, scale),'central',shape);


%Thresholds curve returns the curve with a non negative mask value.


%We calculate the second and third derivate
Second_derivate=Lvv(discgaussfft(inpic, scale), shape);
Third_derivate = Lvvv(discgaussfft(inpic, scale), shape);


%We use zero cross curves to discard the values of lvv with minima zero
%crossing value. 
%Thresholdscurve returns only the curves with a non negative mask value.´
%as zerocrosscurves uses the function thresholdscurve with our second
%paramater as the mask, we have to change lvvv in order that if the value
%of lvvv is less than 0 we get a positive value. This is because we want to
%discard values of lvvv greater than 0 as this are the zero crossing values
%of lvv with minimal value. So, if we get a value of lvvv smaller than
%zero, we change it to a 1, and if it is greater to zero, to minus one. 
Third_Derivate_Mod = 2*((Third_derivate < 0)-0.5);

%We discard the values of lvv with minimal crossing value. 
edgecurves = zerocrosscurves(Second_derivate,Third_Derivate_Mod);

if threshold ~= 0
    %as thresholdcurves discard the curves with mask results smaller than zero, we
    %change our threshold such as every value of the first derivate greater
    %than the threshold is preserved
    First_derivate_mod = ((First_derivate > threshold)-0.5)*2;
else
    %we don't do any thresholding
     First_derivate_mod = First_derivate;
end
%We erase the curves that are don´t pass the threshold. 
out_curves = thresholdcurves(edgecurves,First_derivate_mod);

end