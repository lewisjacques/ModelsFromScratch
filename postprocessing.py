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

        assert(all(isinstance(t,bool) or isinstance(t,np.bool) for t in target)), \
            f"Target values must be boolean. Given:\n\t{target}"
        assert(all(isinstance(p,bool) or isinstance(p,np.bool) for p in predicted)), \
            f"Predicted values must be boolean. Given:\n\t{type(predicted[0])}"
        
        # Calculate the confusion matrix that the metric-functions can query
        self.confusion_matrix = self._get_confusion_matrix(target, predicted)

    @staticmethod
    def _get_confusion_matrix(target:np.array, predicted:np.array) -> np.array:
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

        tp = sum(correct_preds[np.where(predicted)[0]])
        fp = sum(incorrect_preds[np.where(predicted)[0]])
        fn = sum(incorrect_preds[np.where(~predicted)[0]])
        tn = sum(correct_preds[np.where(~predicted)[0]])

        confusion_matrix = np.array([
            [tp,fn],
            [fp,tn]
        ])
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
    
    def model_review(self, ret=False) -> str|None:
        rev = f"""Confusion Matrix:\n{self.confusion_matrix}

        Accuracy: {self.accuracy:.2%}
        Precision: {self.precision:.2%}
        Recall: {self.recall:.2%}
        F1 Score: {self.f1_score:.2%}
        """

        if ret:
            return(rev)
        else: 
            print(rev)