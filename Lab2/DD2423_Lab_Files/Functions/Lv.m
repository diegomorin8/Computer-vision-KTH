function pixels = Lv(inpic,type, shape, show)

if (nargin < 3)
    shape = 'same';
    show = 0;
end
if (nargin < 4)
    show = 0;
end

    Lx = filter2(delta(1,type), inpic, shape);
    Ly = filter2(delta(2,type), inpic, shape);
    pixels = Lx.^2 + Ly.^2;
    if show ~= 0
        figure
        showgrey(pixels)
    end
end
