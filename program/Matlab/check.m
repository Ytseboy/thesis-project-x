function [r] = check(ac,ex)

    r = mean(double(resolveCIM(ac) == resolveCIM(ex)));

end

