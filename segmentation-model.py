
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SegmentationTrainer:

    def __init__(self, train_file: str, target_column: str = 'label'):
        self.train_file = train_file
        self.target_column = target_column
        self.model = None
        self.optimal_k = None
        self.segment_profiles = {}
        self.pca_model = None

        logger.info(f"Initialized SegmentationTrainer")
        logger.info(f"  Train file: {train_file}")

    def load_data_batch(self, batch_size: int = 128):
        for batch in pd.read_csv(self.train_file, chunksize=batch_size):
            yield batch

    def load_full_data(self):
        logger.info("\n   Loading data for segmentation...")
        df = pd.read_csv(self.train_file)

        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        logger.info(f"   Loaded {len(df):,} samples with {len(df.columns)} features")
        return df

    def find_optimal_clusters(self, X, k_range=(2, 10)):
        
        logger.info("FINDING OPTIMAL NUMBER OF CLUSTERS")
        

        inertias = []
        silhouette_scores = []
        k_values = range(k_range[0], k_range[1] + 1)

        for k in k_values:
            logger.info(f"\n   Testing k={k}...")

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouette = silhouette_score(X, labels)
            silhouette_scores.append(silhouette)

            logger.info(f"     Inertia: {kmeans.inertia_:.2f}")
            logger.info(f"     Silhouette Score: {silhouette:.4f}")

        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                second_derivatives.append(second_deriv)

            elbow_idx = np.argmax(second_derivatives) + 1
            elbow_k = list(k_values)[elbow_idx]
        else:
            elbow_k = k_range[0]

        best_silhouette_idx = np.argmax(silhouette_scores)
        best_silhouette_k = list(k_values)[best_silhouette_idx]

        
        logger.info(f"   Elbow method suggests: k={elbow_k}")
        logger.info(f"   Best silhouette score: k={best_silhouette_k} (score={silhouette_scores[best_silhouette_idx]:.4f})")
        

        self.optimal_k = best_silhouette_k

        logger.info(f"\n   Selected optimal k={self.optimal_k}")
        

        return self.optimal_k, inertias, silhouette_scores, list(k_values)

    def train_segmentation(self, n_clusters=None, use_minibatch=False, batch_size=128):
        
        logger.info("TRAINING SEGMENTATION MODEL")
        

        X = self.load_full_data()

        if n_clusters is None:
            logger.info("\n   Finding optimal number of clusters...")
            optimal_k, inertias, silhouette_scores, k_values = self.find_optimal_clusters(X)
            n_clusters = optimal_k

            self._plot_elbow_curve(k_values, inertias, silhouette_scores)
        else:
            self.optimal_k = n_clusters

        logger.info(f"\n   Training K-Means with k={n_clusters}...")

        if use_minibatch:
            self.model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=batch_size,
                n_init=10,
                max_iter=300
            )
            logger.info("Using MiniBatchKMeans")
        else:
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            logger.info("Using KMeans")

        labels = self.model.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)

        logger.info(f"\n   Training complete!")
        logger.info(f"\n   Clustering Metrics:")
        logger.info(f"     Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
        logger.info(f"     Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        logger.info(f"     Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")

        logger.info(f"\n   Segment Sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for seg, count in zip(unique, counts):
            pct = (count / len(labels)) * 100
            logger.info(f"     Segment {seg}: {count:,} samples ({pct:.1f}%)")

        

        self.metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'segment_sizes': dict(zip([int(x) for x in unique], [int(x) for x in counts]))
        }

        return self.model, labels, X

    def analyze_segments(self, X, labels):
        
        logger.info("ANALYZING CUSTOMER SEGMENTS")
        

        df = X.copy()
        df['Segment'] = labels

        for segment in sorted(df['Segment'].unique()):
            
            logger.info(f"SEGMENT {segment} PROFILE")
            

            segment_data = df[df['Segment'] == segment].drop(columns=['Segment'])

            size = len(segment_data)
            pct = (size / len(df)) * 100
            logger.info(f"\nSize: {size:,} customers ({pct:.1f}% of total)")

            logger.info(f"\nTop Distinguishing Features (compared to overall average):")

            overall_mean = X.mean()
            segment_mean = segment_data.mean()

            diff = segment_mean - overall_mean
            diff_pct = (diff / overall_mean) * 100

            sorted_features = diff_pct.abs().sort_values(ascending=False).head(10)

            for feature in sorted_features.index:
                overall_val = overall_mean[feature]
                segment_val = segment_mean[feature]
                diff_val = diff[feature]
                diff_pct_val = diff_pct[feature]

                direction = "higher" if diff_val > 0 else "lower"
                logger.info(f"  - {feature}:")
                logger.info(f"      Segment avg: {segment_val:.2f} | Overall avg: {overall_val:.2f}")
                logger.info(f"      {abs(diff_pct_val):.1f}% {direction} than average")

            self.segment_profiles[segment] = {
                'size': int(size),
                'percentage': float(pct),
                'mean_values': segment_mean.to_dict(),
                'top_features': sorted_features.to_dict()
            }

        

        return self.segment_profiles

    def generate_marketing_insights(self):
        
        logger.info("MARKETING RECOMMENDATIONS BY SEGMENT")
        

        for segment, profile in self.segment_profiles.items():
            
            logger.info(f"SEGMENT {segment} - Marketing Strategy")
            

            size_pct = profile['percentage']

            logger.info(f"\nTarget Audience Size: {profile['size']:,} customers ({size_pct:.1f}%)")

            if size_pct > 30:
                logger.info(f"\n Large segment - Priority for broad marketing campaigns")
            elif size_pct > 15:
                logger.info(f"\n Medium segment - Targeted campaigns recommended")
            else:
                logger.info(f"\n Niche segment - Specialized/personalized marketing")

            logger.info(f"\nKey Characteristics:")
            logger.info(f"  (Review top distinguishing features in segment analysis above)")

            logger.info(f"\nRecommended Actions:")
            logger.info(f"  1. Design targeted messaging based on segment characteristics")
            logger.info(f"  2. Customize product offerings for this segment")
            logger.info(f"  3. Optimize marketing channels for segment preferences")
            logger.info(f"  4. Set appropriate budget allocation (~{size_pct:.1f}% of marketing budget)")

        

    def visualize_segments(self, X, labels, save_dir='segmentation_results'):
        Path(save_dir).mkdir(exist_ok=True)

        logger.info("\n   Generating visualizations...")

        logger.info("\n   Running PCA for visualization...")
        self.pca_model = PCA(n_components=2, random_state=42)
        X_pca = self.pca_model.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Segment')

        if hasattr(self.model, 'cluster_centers_'):
            centers_pca = self.pca_model.transform(self.model.cluster_centers_)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                        c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                        label='Centroids')

        plt.xlabel(f'PC1 ({self.pca_model.explained_variance_ratio_[0] * 100:.1f}% variance)')
        plt.ylabel(f'PC2 ({self.pca_model.explained_variance_ratio_[1] * 100:.1f}% variance)')
        plt.title('Customer Segments (PCA Visualization)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/segments_pca.png', dpi=150)
        plt.close()
        logger.info(f"   Saved segments_pca.png")

        unique, counts = np.unique(labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        bars = plt.bar([f'Segment {s}' for s in unique], counts, color='steelblue', edgecolor='black')
        plt.ylabel('Number of Customers')
        plt.title('Segment Size Distribution')

        total = len(labels)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{count:,}\n({pct:.1f}%)',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/segment_sizes.png', dpi=150)
        plt.close()
        logger.info(f"   Saved segment_sizes.png")

        if self.segment_profiles:
            n_segments = len(self.segment_profiles)
            n_top_features = 10

            feature_data = []
            feature_names = []

            for segment in sorted(self.segment_profiles.keys()):
                top_features = self.segment_profiles[segment]['top_features']
                if not feature_names:
                    feature_names = list(top_features.keys())[:n_top_features]

                feature_values = [top_features.get(f, 0) for f in feature_names]
                feature_data.append(feature_values)

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                np.array(feature_data).T,
                xticklabels=[f'Segment {i}' for i in range(n_segments)],
                yticklabels=feature_names,
                cmap='RdYlGn',
                center=0,
                annot=True,
                fmt='.1f',
                cbar_kws={'label': '% Difference from Average'}
            )
            plt.title('Segment Feature Profiles (% Difference from Overall Average)')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/segment_features_heatmap.png', dpi=150)
            plt.close()
            logger.info(f"   Saved segment_features_heatmap.png")

        logger.info(f"\n   All visualizations saved to {save_dir}/")

    def _plot_elbow_curve(self, k_values, inertias, silhouette_scores):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(k_values, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(alpha=0.3)

        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score by k')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('segmentation_results/elbow_analysis.png', dpi=150)
        plt.close()
        logger.info(f"   Saved elbow_analysis.png")

    def save_model(self, output_dir='models'):
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = f'{output_dir}/segmentation_model_{timestamp}.joblib'
        joblib.dump({
            'model': self.model,
            'pca_model': self.pca_model,
            'optimal_k': self.optimal_k
        }, model_path)
        logger.info(f"\n   Model saved: {model_path}")

        profiles_path = f'{output_dir}/segment_profiles_{timestamp}.json'
        with open(profiles_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'profiles': self.segment_profiles
            }, f, indent=2)
        logger.info(f"   Profiles saved: {profiles_path}")

        logger.info(f"Model and profiles saved to {output_dir}/")

        return model_path, profiles_path

    def predict_segment(self, X):
        if self.model is None:
            raise ValueError("No model trained")
        return self.model.predict(X)


if __name__ == "__main__":
    
    logger.info("CUSTOMER SEGMENTATION MODEL TRAINING")
    

    trainer = SegmentationTrainer(
        train_file='preprocessed_data_train.csv',
        target_column='label'
    )

    model, labels, X = trainer.train_segmentation(n_clusters=None, use_minibatch=False)

    segment_profiles = trainer.analyze_segments(X, labels)

    trainer.generate_marketing_insights()

    trainer.visualize_segments(X, labels)

    trainer.save_model()

    
    logger.info("SEGMENTATION TRAINING COMPLETE")
    