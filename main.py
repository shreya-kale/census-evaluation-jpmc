import logging
from pathlib import Path
from datetime import datetime
import sys

from data_preprocessing import DataLoader, DataExplorer, DataPreprocessor
from classification_model import ClassificationTrainer
from segmentation_model import SegmentationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProjectPipeline:

    def __init__(self, data_file: str, columns_file: str):
        self.data_file = data_file
        self.columns_file = columns_file
        self.results = {}

        Path('outputs').mkdir(exist_ok=True)

        logger.info("CENSUS INCOME CLASSIFICATION AND SEGMENTATION PROJECT")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Data File: {data_file}")
        logger.info(f"Columns File: {columns_file}")

    def run_preprocessing(self, batch_size=128):
        logger.info("PHASE 1: DATA PREPROCESSING")

        try:
            loader = DataLoader(
                data_file=self.data_file,
                columns_file=self.columns_file,
                batch_size=batch_size
            )

            explorer = DataExplorer(loader)
            insights = explorer.explore(n_batches=20)
            explorer.visualize()

            logger.info("Starting preprocessing...")
            preprocessor = DataPreprocessor(
                loader=loader,
                target_column=insights['target_column']
            )

            preprocessor.compute_statistics()

            train_file, test_file = preprocessor.preprocess_and_save(
                output_file='preprocessed_data.csv',
                drop_cols=insights['missing_cols'] if len(insights['missing_cols']) > 5 else None,
                fill_missing='mean',
                normalize=True,
                train_split=0.8
            )

            preprocessor.save_preprocessing_config()

            self.results['preprocessing'] = {
                'train_file': train_file,
                'test_file': test_file,
                'target_column': insights['target_column'],
                'insights': insights
            }

            logger.info("Phase 1 Complete: Data preprocessing successful")
            return True

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return False

    def run_classification(self, model_type='sgd'):
        logger.info("PHASE 2: CLASSIFICATION MODEL TRAINING")

        try:
            train_file = self.results['preprocessing']['train_file']
            test_file = self.results['preprocessing']['test_file']
            target_column = self.results['preprocessing']['target_column']

            trainer = ClassificationTrainer(
                train_file=train_file,
                test_file=test_file,
                target_column=target_column
            )

            if model_type == 'sgd':
                model = trainer.train_incremental(model_type='sgd', batch_size=128)
            else:
                model = trainer.train_full(model_type='random_forest')

            metrics, y_test, y_pred, y_pred_proba = trainer.evaluate()

            trainer.visualize_results(y_test, y_pred, y_pred_proba)

            model_path, metrics_path = trainer.save_model()

            self.results['classification'] = {
                'model_path': model_path,
                'metrics': metrics
            }

            logger.info("Phase 2 Complete: Classification model trained and evaluated")
            return True

        except Exception as e:
            logger.error(f"Classification training failed: {str(e)}")
            return False

    def run_segmentation(self, n_clusters=None):
        logger.info("PHASE 3: CUSTOMER SEGMENTATION")

        try:
            train_file = self.results['preprocessing']['train_file']
            target_column = self.results['preprocessing']['target_column']

            trainer = SegmentationTrainer(
                train_file=train_file,
                target_column=target_column
            )

            model, labels, X = trainer.train_segmentation(
                n_clusters=n_clusters,
                use_minibatch=False
            )

            segment_profiles = trainer.analyze_segments(X, labels)

            trainer.generate_marketing_insights()

            trainer.visualize_segments(X, labels)

            model_path, profiles_path = trainer.save_model()

            self.results['segmentation'] = {
                'model_path': model_path,
                'profiles_path': profiles_path,
                'n_clusters': trainer.optimal_k,
                'metrics': trainer.metrics
            }

            logger.info("Phase 3 Complete: Segmentation model created and analyzed")
            return True

        except Exception as e:
            logger.error(f"Segmentation training failed: {str(e)}")
            return False

    def generate_summary(self):
        logger.info("PROJECT EXECUTION SUMMARY")

        logger.info("1. PREPROCESSING")
        logger.info(f"   Train file: {self.results['preprocessing']['train_file']}")
        logger.info(f"   Test file: {self.results['preprocessing']['test_file']}")
        logger.info(f"   Target column: {self.results['preprocessing']['target_column']}")

        logger.info("2. CLASSIFICATION RESULTS")
        metrics = self.results['classification']['metrics']
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"   Model saved: {self.results['classification']['model_path']}")

        logger.info("3. SEGMENTATION RESULTS")
        seg_metrics = self.results['segmentation']['metrics']
        logger.info(f"   Number of segments: {self.results['segmentation']['n_clusters']}")
        logger.info(f"   Silhouette Score: {seg_metrics['silhouette_score']:.4f}")
        logger.info(f"   Model saved: {self.results['segmentation']['model_path']}")
        logger.info(f"   Profiles saved: {self.results['segmentation']['profiles_path']}")

        logger.info("4. OUTPUT FILES GENERATED")
        logger.info("   Exploration:")
        logger.info("     - plots/target_distribution.png")
        logger.info("     - plots/correlation.png")
        logger.info("   Classification:")
        logger.info("     - classification_results/confusion_matrix.png")
        logger.info("     - classification_results/roc_curve.png")
        logger.info("     - classification_results/metrics_comparison.png")
        logger.info("   Segmentation:")
        logger.info("     - segmentation_results/segments_pca.png")
        logger.info("     - segmentation_results/segment_sizes.png")
        logger.info("     - segmentation_results/segment_features_heatmap.png")
        logger.info("     - segmentation_results/elbow_analysis.png")

        logger.info(f"PROJECT COMPLETED SUCCESSFULLY")
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def run_full_pipeline(self, classification_model='sgd', n_clusters=None):
        logger.info("Executing full ML pipeline...")

        success = self.run_preprocessing(batch_size=128)
        if not success:
            logger.error("Pipeline failed at preprocessing stage")
            sys.exit(1)

        success = self.run_classification(model_type=classification_model)
        if not success:
            logger.error("Pipeline failed at classification stage")
            sys.exit(1)

        success = self.run_segmentation(n_clusters=n_clusters)
        if not success:
            logger.error("Pipeline failed at segmentation stage")
            sys.exit(1)

        self.generate_summary()

        return self.results


if __name__ == "__main__":
    DATA_FILE = 'file_name.data'
    COLUMNS_FILE = 'file_name.columns'

    pipeline = ProjectPipeline(
        data_file=DATA_FILE,
        columns_file=COLUMNS_FILE
    )

    results = pipeline.run_full_pipeline(
        classification_model='sgd',
        n_clusters=None
    )

    logger.info("All results saved. Check the following directories:")
    logger.info("  - plots/ (data exploration)")
    logger.info("  - classification_results/ (classification outputs)")
    logger.info("  - segmentation_results/ (segmentation outputs)")
    logger.info("  - models/ (saved models)")
    logger.info("  - artifacts/ (preprocessing configs)")