function [y, minvalue, c] = minmaxnor(x)
y = (x - min(x)) ./ (max(x)-min(x));
minvalue = min(x);
c = max(x) - min(x);
end