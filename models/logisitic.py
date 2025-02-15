class LogisticModel:
    def __init__(self):
        return
    
# z = W.X + b
# yhat = sigmoid_function(z) = 1 / (1 + e^(-z))

# An epoch is one complete pass through the entire training dataset during training.
# 	•	The model updates its weights once per batch in each epoch.
# 	•	More epochs help the model learn better, but too many can lead to overfitting (memorizing instead of generalizing).
# 	•	A common strategy is to monitor the loss and stop training when it stops improving (early stopping).

# Bias allows the decision boundary to shift for better fitting.
#   - Effectively the constant in the linear equation model
#   - we want to assure y=WX doesn't pass through the origin


# Use gradient decent to minimise the cost function
#   alpha = learning-rate
#   Each iteration:
#       W = W - (alpha * dL/dW)
#       b = b - (alpha * dL/db)

# Gradient calculation (derived by the chain rule)
# dL/dW = 1/n sum(i=1:N) { X^T.(yhati - yi) }
# dL/db = 1/n sum(i=1:N) { (yhati - yi)