import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

def evaluate_all_models():
    print("Loading final multi-model telemetry data...")
    df = pd.read_csv("final_multimodel_telemetry.csv")
    y_true = df['label'] 
    
    models = {
        'Z-Score (Statistical)': 'zscore_anomaly',
        'Isolation Forest (ML)': 'if_anomaly',
        'LSTM Autoencoder (DL)': 'lstm_anomaly',
        'Consensus Engine (Final)': 'severe_anomaly'
    }
    
    results = []
    
    # -----------------------------------------
    # 1. GENERATE CONFUSION MATRIX VISUALS
    # -----------------------------------------
    print("Generating Confusion Matrix Visuals...")
    fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (model_name, col_name) in enumerate(models.items()):
        y_pred = df[col_name]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics for the CSV
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        
        results.append({
            'Model': model_name,
            'Accuracy (%)': round(acc, 2),
            'Precision (%)': round(prec, 2),
            'Recall (%)': round(rec, 2),
            'True Positives (TP)': tp,
            'False Positives (FP)': fp,
            'True Negatives (TN)': tn,
            'False Negatives (FN)': fn
        })
        
        # Plot individual confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'],
                    annot_kws={"size": 14, "weight": "bold"})
        
        axes[i].set_title(model_name, fontsize=14, fontweight='bold', pad=10)
        axes[i].set_ylabel('Actual (Ground Truth)', fontsize=12)
        axes[i].set_xlabel('Predicted by AI', fontsize=12)

    plt.tight_layout()
    plt.savefig('confusion_matrices_grid.png', dpi=300)
    print("✅ Visual Confusion Matrix saved as 'confusion_matrices_grid.png'")
    
    # -----------------------------------------
    # 2. EXPORT RESULTS TO CSV
    # -----------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("✅ Metrics saved to 'evaluation_results.csv'")
    
    # -----------------------------------------
    # 3. GENERATE BAR CHART
    # -----------------------------------------
    print("Generating Performance Bar Chart...")
    plt.figure(figsize=(11, 6))
    bar_width = 0.35
    r1 = np.arange(len(results_df))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, results_df['Precision (%)'], color='#2c3e50', width=bar_width, edgecolor='white', label='Precision')
    plt.bar(r2, results_df['Recall (%)'], color='#3498db', width=bar_width, edgecolor='white', label='Recall')
    
    plt.xlabel('Detection Models', fontweight='bold', fontsize=12)
    plt.ylabel('Percentage Score (%)', fontweight='bold', fontsize=12)
    plt.xticks([r + bar_width/2 for r in range(len(results_df))], results_df['Model'], rotation=10)
    plt.title('Model Performance Comparison: Precision vs Recall', fontweight='bold', fontsize=14)
    
    for i in range(len(r1)):
        plt.text(r1[i], results_df['Precision (%)'][i] + 1.5, f"{results_df['Precision (%)'][i]}%", ha='center', fontsize=10, fontweight='bold')
        plt.text(r2[i], results_df['Recall (%)'][i] + 1.5, f"{results_df['Recall (%)'][i]}%", ha='center', fontsize=10, fontweight='bold')
        
    plt.ylim(0, 115) 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=300)
    print("✅ Bar chart saved as 'model_comparison_chart.png'")
    print("="*55)

if __name__ == "__main__":
    evaluate_all_models()
    