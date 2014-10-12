function f = makeAnswer(Theta1, Theta2)

	data = load('../testForKaggle.csv');
	A = predict(Theta1, Theta2, data) - 1;
	A = [(1:28000)' A];

	csvwrite('kaggleAnswer.csv',A)
	f = 'done Answer';
end	