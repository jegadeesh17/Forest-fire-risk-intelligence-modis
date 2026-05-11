from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_model(y_test, y_pred, model_name="Model"):
    """Print classification report and ROC-AUC score."""
    print(f"--- {model_name} Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    try:
        auc = roc_auc_score(y_test, y_pred)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC could not be calculated (possibly only one class in y_test).")
    
    return confusion_matrix(y_test, y_pred)
