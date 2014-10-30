%% Confusion matrix for multinomial classifier
function C = conMatrix(ac, ex)

	q = length(unique(ex));
	C = zeros(q,q);

	for i = 1:q
		for k = 1:q
			C(i,k) = sum(ex==i & ac == k);
		endfor
	endfor

end