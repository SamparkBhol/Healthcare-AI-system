from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class EvaluationMetrics:
    def __init__(self):
        pass

    def evaluate_classification(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average="weighted")
        recall = recall_score(true_labels, predicted_labels, average="weighted")
        f1 = f1_score(true_labels, predicted_labels, average="weighted")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    def evaluate_regression(self, true_values, predicted_values):
        mse = np.mean((true_values - predicted_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_values - predicted_values))

        return {"mse": mse, "rmse": rmse, "mae": mae}

    def custom_healthcare_metric(self, true_outcomes, predicted_outcomes):
        # Custom metric to evaluate treatment efficacy or other healthcare metrics
        effectiveness = np.mean(predicted_outcomes == true_outcomes)
        return {"effectiveness": effectiveness}

if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    true_labels = [1, 0, 1, 1, 0]
    predicted_labels = [1, 0, 0, 1, 1]

    classification_metrics = evaluator.evaluate_classification(true_labels, predicted_labels)
    print(f"Classification metrics: {classification_metrics}")

    true_values = np.array([3.5, 2.0, 4.0])
    predicted_values = np.array([3.6, 1.8, 3.9])

    regression_metrics = evaluator.evaluate_regression(true_values, predicted_values)
    print(f"Regression metrics: {regression_metrics}")
