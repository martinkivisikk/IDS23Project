import matplotlib.pyplot as plt
import seaborn as sns


def plot_probability_distribution(model, X_test, y_test, model_name):
    # Get predicted probabilities for the positive class
    y_pred_proba_positive = model.predict_proba(X_test)[:, 1]

    # Separate the predicted probabilities based on the true class
    y_pred_proba_positive_true = y_pred_proba_positive[y_test == 1]
    y_pred_proba_positive_false = y_pred_proba_positive[y_test == 0]

    # Plot the probability distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba_positive_true, kde=True, color='blue', label='True Class (Spam)')
    sns.histplot(y_pred_proba_positive_false, kde=True, color='red', label='True Class (Non-Spam)')

    plt.title(f'Probability Distribution - {model_name}')
    plt.xlabel('Predicted Probability of Positive Class (Spam)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
