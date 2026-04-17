import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    """
    Evaluates the trained model on test data and prints out training and testing accuracy, 
    along with a classification report and a confusion matrix plot.
    """
    print("\n--- Evaluation Results ---")
    
    # 1. Training Accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {(train_acc * 100):.2f}%")
    
    # 2. Testing Accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy:  {(test_acc * 100):.2f}%\n")
    
    print("Classification Report (Testing Data):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # 3. Create a clean, professional Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    
    # Plotting heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                cbar_kws={'label': 'Number of Images'},
                xticklabels=class_names, yticklabels=class_names)
                
    plt.title(f'Logistic Regression Results\nTraining Acc: {train_acc*100:.1f}% | Testing Acc: {test_acc*100:.1f}%\n', fontsize=14)
    plt.ylabel('Actual Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_results.png', dpi=300)
    print("Graph successfully saved to 'confusion_matrix_results.png' for presentation!")
    
    return train_acc, test_acc
