function g = sigmoidGradient(z)
%returns the gradient of the sigmoid function

g = sigmoid(z).*(1 - sigmoid(z));

end
