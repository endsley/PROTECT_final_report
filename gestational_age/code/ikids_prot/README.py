

data 1 (Most basic setting)
	1. pick most important features and impute the missing ones
	2. Did not deal with imbalance

	Regression
		1. Ran bunch of basic regressors, did not perform well
		2. Ran large neural network regression, performed okay, 
			the training did very well, but the test had large worse cases and does not seem to predict premature
			this suggests that test results might be improved if 
				1. we add more regularizes from prior assumptions it might perform better
				2. First balance the data better might help
				3. Use a different way in place of imputation (like autoencoder)

			def costFunction(x, y, ŷ, ind):
				return torch.sum((y- ŷ) ** 2)

			def costFunction2(x, y, ŷ, ind):
				n = x.shape[0]
				relu = nn.ReLU()
				penalty = (torch.sum(relu((ŷ - y)))*torch.sum(relu((37-y))))/(n*n)
				return torch.sum((y- ŷ) ** 2) + penalty

		3. Ran autoencoder regressor (as a regularizer), did not do well 
			def costFunction(x, x̂, ẙ, y, ŷ, ind):	
				n  = x.shape[0]
				λ = 0.1
				R = torch.sum((x - ŷ) ** 2)/(32*n)	#scaled by batch size times data dimension
				R2 = torch.sum((ẙ - y) ** 2)/(n)
				loss = R + R2
				return [loss, R, R2]

			def costFunction3(x, x̂, ẙ, y, ŷ, ind):	
				n  = x.shape[0]
				relu = nn.ReLU()
				R = torch.sum((x - ŷ) ** 2)/(32*n)	#scaled by batch size times data dimension
				R2 = 5*torch.sum((ẙ - y) ** 2)/(n)
				penalty = 0.01*(torch.sum(relu((ẙ - y))))/(n)
				loss = R + R2 + penalty
				return [loss, R, R2]

	Classifiction
		1. (Not yet) Use CE  Accuracy: 87%, Precision: 87%, Recall: 87%


Data 2 
	1. Use classification labels to try to balance the data with "resampling" of smaller class + impute

	Regression
		1. (Not yet) Ran large neural network regression
		2. (Not yet) Ran autoencoder regressor

	Classifiction
		1. Use CE : Performed well

	Regression + Classification
		2. What if we use both objective acting as constraints?

Data 3 
	1. Use classification labels to try to balance the data with "smote" of smaller class

	Classifiction and Regression together with Autoencoder, 
	so far it has the best results

    Classifier Accuracy : 0.9913
    Classifier Precision: 0.9811
    Classifier Recall: 1.0000
    Regressor accuracy: 0.9898
    Regressor Precision: 0.9781
    Regressor Recall: 1.0000
    Mean absolute error: 0.7375	
