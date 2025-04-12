from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_scores(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.2f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='macro'):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='macro'):.2f}")
