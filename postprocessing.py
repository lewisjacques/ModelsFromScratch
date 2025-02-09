import numpy as np

# Implement performance metrics
class PostProcess:
    def __init__(self, target:np.array, predicted:np.array):
        """
        Handle building up all performance metrics into one class

        Args:
            target (np.array): list of boolean actual values
            predicted (np.array): list of boolean predicted values
        """

        assert(all(isinstance(t,bool) for t in target)), \
            "Target values must be boolean"
        assert(all(isinstance(p,bool) for p in predicted)), \
            "Predicted values must be boolean"
        
        # Calculate the confusion matrix that the metric-functions can query
        self.confusion_matrix = self._get_confusion_matrix(target, predicted)

    def _get_confusion_matrix(self, target:np.array, predicted:np.array) -> np.array:
        """
        Build the confusion matrix based on the predicted and target variables
        provided

        Actual/Predicted 1 0
        1     TP          FN
        0     FP          TN

        Args:
            target (np.array): list of boolean actual values
            predicted (np.array): list of boolean predicted values

        Returns:
            np.array: 2D confusion matrix array
        """

        correct_preds = target==predicted
        incorrect_preds = target!=predicted

        confusion_matrix = np.zeros((2,2))
        # True positive
        confusion_matrix[0][0] = sum(correct_preds[np.where(predicted)])
        # False positive
        confusion_matrix[1][0] = sum(incorrect_preds[np.where(predicted)])
        # False negative
        confusion_matrix[0][1] = sum(incorrect_preds[np.where(~predicted)])
        # True negative
        confusion_matrix[1][1] = sum(correct_preds[np.where(~predicted)])
        return(confusion_matrix)

    @property
    def accuracy(self) -> float:
        """Computes Accuracy: (TP + TN) / (TP + TN + FP + FN)
        Proportion of correct predictions"""
        cf = self.confusion_matrix
        return (cf[0, 0] + cf[1, 1]) / np.sum(cf)

    @property
    def precision(self) -> float:
        """Computes Precision: TP / (TP + FP)
        Proportion of correct positive predictions"""
        cf = self.confusion_matrix
        return cf[0, 0] / (cf[0, 0] + cf[1, 0]) if (cf[0, 0] + cf[1, 0]) > 0 else 0.0

    @property
    def recall(self) -> float:
        """Computes Recall: TP / (TP + FN)
        Proportion of true values that were correctly predicted"""
        cf = self.confusion_matrix
        return cf[0, 0] / (cf[0, 0] + cf[0, 1]) if (cf[0, 0] + cf[0, 1]) > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Computes F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        Balances both precision and recall"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0