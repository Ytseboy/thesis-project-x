%Function assert to measure Actual vs Expected Percentage
function p = assert(Theta1, Theta2, X, y)

	actual = predict(Theta1, Theta2, X);
	p = mean(double( actual == y )) * 100;

end