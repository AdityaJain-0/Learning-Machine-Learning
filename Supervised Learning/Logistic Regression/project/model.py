import numpy as np

class LogisticRegression: 
    #def the properties and features of the model
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None

    #def the activation function 
    def _sigmoid(self, z):
        return 1/ (1 + np.exp(-z))

    #Train the model
    def fit(self, X, y):

        #The matrix of samples and features is the same as the training data
        num_samples, num_features = X.shape

        #Set the intial feature data 0, weights is a matrix of 0s because the formula is w1*x1, but all the weigghts will be multiplied against the all the x's at the same time and then checked iterativly
        self.weights = np.zeros(num_features)
        self.bias = 0

        for iter in range(self.iterations):

            #Def of logistic regression
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            #gradient 
            dw = (1/ num_samples) * np.dot(X.T, (y_predicted - y))

            db = (1 / num_samples) * np.sum(y_predicted - y)

            #Update features 
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

            # print(self.weights, self.bias)
            # if iter == 1:
            #      print(linear_model, y_predicted, X, y, dw, db, self.weights, self.bias)

    def predict_prob(self, X):
            linear_model = np.dot(X, self.weights) + self.bias
            return self._sigmoid(linear_model)
    
    def predict(self, X):
         #Used to test training data
         probas = self.predict_prob(X)
         return [1 if i >= 0.5 else 0 for i in probas]
        