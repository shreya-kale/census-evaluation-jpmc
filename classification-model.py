
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClassificationTrainer:
    

    def __init__(self, train_file: str, test_file: str, target_column: str = 'label'):
        self.train_file = train_file
        self.test_file = test_file
        self.target_column = target_column
        self.model = None
        self.metrics = {}

        logger.info(f"Initialized ClassificationTrainer")
        logger.info(f"  Train file: {train_file}")
        logger.info(f"  Test file: {test_file}")
        logger.info(f"  Target: {target_column}")

    def load_data(self, file_path: str, batch_size: int = 128):
        
        for batch in pd.read_csv(file_path, chunksize=batch_size):
            yield batch

    def train_incremental(self, model_type: str = 'sgd', batch_size: int = 128):
        
        print("TRAINING CLASSIFICATION MODEL (INCREMENTAL)")
        

        # Initialize model
        if model_type == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',  # Logistic regression
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5
            )
            logger.info("Using SGDClassifier with logistic loss")
        else:
            raise ValueError(f"Model type {model_type} not supported for incremental training")

        # Get unique classes first
        print("\n   Scanning for unique classes...")
        classes = self._get_classes()
        print(f"   Found {len(classes)} classes: {classes}")

        # Train incrementally
        print("\n   Training on batches...")
        batch_count = 0
        total_samples = 0

        for batch in self.load_data(self.train_file, batch_size):
            if self.target_column not in batch.columns:
                logger.error(f"Target column '{self.target_column}' not found in data")
                raise ValueError(f"Target column '{self.target_column}' not found")

            X = batch.drop(columns=[self.target_column])
            y = batch[self.target_column]

            # Train on batch
            self.model.partial_fit(X, y, classes=classes)

            batch_count += 1
            total_samples += len(batch)

            if batch_count % 100 == 0:
                print(f"   Processed {batch_count} batches ({total_samples} samples)...")

        print(f"\n   Training complete!")
        print(f"   Total batches: {batch_count}")
        print(f"   Total samples: {total_samples:,}")
        

        return self.model

    def train_full(self, model_type: str = 'random_forest'):
        
        print("TRAINING CLASSIFICATION MODEL (FULL DATA)")
        

        # Load full training data
        print("\n   Loading training data...")
        train_df = pd.read_csv(self.train_file)

        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]

        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {len(X_train.columns)}")

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )
            logger.info("Using RandomForestClassifier")
        elif model_type == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',
                max_iter=1000,
                random_state=42
            )
            logger.info("Using SGDClassifier")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        print("\n   Training model...")
        self.model.fit(X_train, y_train)

        print(f"   Training complete!")
        

        return self.model

    def evaluate(self):
        
        
        print("EVALUATING MODEL")
        

        if self.model is None:
            raise ValueError("No model trained. Call train_incremental() or train_full() first.")

        # Load test data
        print("\n   Loading test data...")
        test_df = pd.read_csv(self.test_file)

        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]

        print(f"   Test samples: {len(X_test):,}")

        # Predictions
        print("\n   Making predictions...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }

        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        # Print results
        
        print("PERFORMANCE METRICS")
        
        print(f"   Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"   Precision: {self.metrics['precision']:.4f}")
        print(f"   Recall:    {self.metrics['recall']:.4f}")
        print(f"   F1-Score:  {self.metrics['f1']:.4f}")
        if 'roc_auc' in self.metrics:
            print(f"   ROC-AUC:   {self.metrics['roc_auc']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print("CONFUSION MATRIX")
        
        print(cm)

        # Classification report
        
        print("CLASSIFICATION REPORT")
        
        print(classification_report(y_test, y_pred, zero_division=0))

        

        return self.metrics, y_test, y_pred, y_pred_proba

    def _get_classes(self):
        
        classes = set()
        for batch in self.load_data(self.train_file, batch_size=1000):
            classes.update(batch[self.target_column].unique())
        return np.array(sorted(classes))

    def visualize_results(self, y_test, y_pred, y_pred_proba=None, save_dir: str = 'classification_results'):
        
        Path(save_dir).mkdir(exist_ok=True)

        logger.info("\n   Generating visualizations...")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150)
        plt.close()
        logger.info(f" Saved confusion_matrix.png")

        # 2. ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {self.metrics['roc_auc']:.4f})")
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/roc_curve.png', dpi=150)
            plt.close()
            logger.info(f"Saved roc_curve.png")

        # 3. Metrics Bar Chart
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        values = [self.metrics[m] for m in metrics_to_plot]
        bars = plt.bar(metrics_to_plot, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('Classification Metrics')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=150)
        plt.close()
        logger.info(f"Saved metrics_comparison.png")

        logger.info(f"\n   All visualizations saved to {save_dir}/")

    def save_model(self, output_dir: str = 'models'):
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = f'{output_dir}/classification_model_{timestamp}.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"\n   Model saved: {model_path}")

        # Save metrics
        metrics_path = f'{output_dir}/classification_metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"   Metrics saved: {metrics_path}")

        logger.info(f"Model and metrics saved to {output_dir}/")

        return model_path, metrics_path

    def load_model(self, model_path: str):
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model


# Usage
if __name__ == "__main__":
    
    logger.info("INCOME CLASSIFICATION MODEL TRAINING")
    

    # Initialize trainer
    trainer = ClassificationTrainer(
        train_file='preprocessed_data_train.csv',
        test_file='preprocessed_data_test.csv',
        target_column='label'
    )

    # Choose training approach based on data size
    logger.info("\nSelect training approach:")
    logger.info("  1. Incremental (for large datasets, uses SGD)")
    logger.info("  2. Full data (Random Forest, assumes data fits in memory)")

    # For this example, let's use incremental training (production-ready)
    choice = 1  # Change to 2 for Random Forest

    if choice == 1:
        # Incremental training (scalable)
        model = trainer.train_incremental(model_type='sgd', batch_size=128)
    else:
        # Full training (if data fits in memory)
        model = trainer.train_full(model_type='random_forest')

    # Evaluate
    metrics, y_test, y_pred, y_pred_proba = trainer.evaluate()

    # Visualize
    trainer.visualize_results(y_test, y_pred, y_pred_proba)

    # Save model
    trainer.save_model()

    
    logger.info("CLASSIFICATION TRAINING COMPLETE")
    