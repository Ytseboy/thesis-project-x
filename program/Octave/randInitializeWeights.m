function W = randInitializeWeights(L_in, L_out)

%Randomly initialize the weights
%L_in = input, out = output weights
e = 0.12;
W = rand(L_out, 1 + L_in) * 2 * e - e; % +1 stands for bias unit

end
