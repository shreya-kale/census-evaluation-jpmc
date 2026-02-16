# Necessary file imports
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Dict, Any
import seaborn as sns

# Supress 'backward_compatibility' warnings in Pandas
import warnings
warnings.filterwarnings('ignore', message='.*backward compatibility.*')

# File logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO,
                    handlers=[logging.FileHandler('logs/data-preprocessing.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self, data_file: str, columns_file:str, batch_size: int = 128) -> None:
        self.data_file = Path(data_file)
        self.columns_file = Path(columns_file)
        self.batch_size = batch_size

        # Files validation
        self.validate_files()

        # Load columns file
        self.columns = self.load_columns()

        # Logger entry
        logger.info(f"Initialized: Loaded columns {self.columns}, batch_size={batch_size}")

    def validate_files(self) -> None:

        # Validate that required files exist in given location
        if not self.data_file.exists():
            raise FileNotFoundError(f"File {self.data_file} does not exist")
        if not self.columns_file.exists():
            raise FileNotFoundError(f"File {self.columns_file} does not exist")

    def load_columns(self):

        # Load column names from columns_file and validate names
        try:
            with open(self.columns_file, 'r') as f:
                columns = [line.strip() for line in f.readlines()]

            if not columns:
                raise ValueError("No columns provided in {self.columns_file}")

            logger.info(f"Loaded {len(columns)} columns from {self.columns_file}")
            return columns

        except Exception as e:
            logger.error(f"Failed to load {self.columns_file}: {e}")
            raise e

    def load_batches(self) -> Iterator[pd.DataFrame]:

        # Generate batches of data with chunk_size: 128
        try:
            batch_num = 0
            total_rows = 0

            for batch in pd.read_csv(self.data_file, names=self.columns, header=None, delimiter=',', chunksize=self.batch_size, on_bad_lines='warn'):
                batch_num += 1
                total_rows += len(batch)

                # Validate batch entry
                if not self.validate_batch(batch, batch_num):
                    logger.warning(f"Batch {batch_num} of {self.data_file} failed. Skipping invalid batch")
                    continue

                yield batch

            logger.info(f"Batch {batch_num} of {self.data_file} complete")

        except Exception as e:
                logger.error(f"Failed to load batches: {e}")
                raise e

    def validate_batch(self, batch: pd.DataFrame, batch_num: int) -> bool:

        # Validate batch data
        # Check if batch is empty
        if len(batch) == 0:
            logger.warning(f"Batch {batch_num} of {self.data_file} has no data")
            return False

        # Check if all columns are present
        if len(batch.columns) != len(self.columns):
            logger.error(f"Batch {batch_num} has {len(batch.columns)} columns, expected {len(self.columns)}")
            return False

        return True

    def get_sample(self, n_batches: int=5) -> pd.DataFrame:

        # Create samples of data for exploration
        logger.info(f"Collecting {n_batches} batches for sampling")
        samples = []

        for i, batch in enumerate(self.load_batches()):
            samples.append(batch)
            if i + 1 >= n_batches:
                break

        sample_df = pd.concat(samples, ignore_index=True)
        logger.info(f"Sample of {len(sample_df)} size created")

        return sample_df


class DataExplorer:

    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader
        self.sample_df = None

    def explore(self, n_batches: int = 20) -> Dict[str, Any]:
        logger.info(f"Data explorer started")

        # Get data sample
        self.sample_df = self.data_loader.get_sample(n_batches=n_batches)
        logger.info(f"Sample obtained")

        # Dataset Overview
        logger.info("Dataset Overview")
        logger.info(f"\tRows: {len(self.sample_df)}")
        logger.info(f"\tColumns: {len(self.sample_df.columns)}")
        logger.info(f"\tMemory: {self.sample_df.memory_usage(index=True, deep=True).sum() / (1024 ** 2):.2f} MB\n")

        # Type of data in dataset
        logger.info("Data Types")
        numeric_cols = self.sample_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.sample_df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}\n")

        # Identifying Missing Values
        logger.info("Missing Values")
        missing_values = self.sample_df.isnull().sum()
        logger.info(f"Missing values:\n{missing_values}")
        missing_cols = missing_values[missing_values > 0]
        if len(missing_cols) > 0:
            missing_pct = (missing_cols / len(self.sample_df)) * 100
            logger.info(f"Missing Column Values: {len(missing_cols)}")
            for col in missing_cols.index[:5]:
                logger.info(f"\t{col}: {missing_values[col]} ({(missing_values[col] / len(self.sample_df)) * 100:.1f}%)\n")
        else:
            logger.info("No missing values\n")

        # Target Variable (Label Column)
        logger.info("Target Variable Analysis")
        target_col = self.sample_df.columns[-1]
        n_unique = self.sample_df[target_col].nunique()
        logger.info(f"Column name: {target_col}")
        logger.info(f"Unique values: {n_unique}")
        logger.info(f"Distribution:\n{self.sample_df[target_col].value_counts()}\n")

        # Data Statistics
        logger.info("Data Statistics")
        logger.info("Numeric Columns Description")
        logger.info(self.sample_df[numeric_cols].describe())
        logger.info("Categorical Columns Description")
        logger.info(self.sample_df[categorical_cols].describe())
        logger.info("Description Ready\n")

        logger.info("Exploration insights ready. Function returns Dict[str, Any]")

        return {
            'target_column': target_col,
            'n_classes': n_unique,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'missing_cols': missing_cols.index.tolist()
        }

    def visualize(self, save_dir: str='plots'):

        # Generate necessary plots and visual aids
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.sample_df is None:
            logger.warning(f"No data found, execute explore() first")
            return

        # Target Value Distribution
        target_col = self.sample_df.columns[-1]
        if self.sample_df[target_col].nunique() <20:
            plt.figure(figsize=(10, 6))
            self.sample_df[target_col].value_counts().plot(kind='bar')
            plt.title(f"Target Distribution: {target_col}")
            plt.xlabel(target_col)
            plt.ylabel(f"Count")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'target_distribution.png'), dpi=150)
            plt.close()
            logger.info(f"Target Distribution saved to {os.path.join(save_dir, 'target_distribution.png')}")

        # Numeric Columns Correlation Heatmap
        numeric_df = self.sample_df.select_dtypes(include=[np.number]).copy()
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0.5)
            plt.title(f"Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=150)
            plt.close()
            logger.info(f"Correlation Heatmap saved to {os.path.join(save_dir, 'correlation_heatmap.png')}")
            logger.info("Data visualization complete")


class DataPreprocessor:
    # Preprocess data

    def __init__(self, loader: DataLoader, target_column: str = 'label'):
        self.loader = loader
        self.target_column = target_column
        self.stats = {}
        self.columns_to_drop = []

    def compute_statistics(self):
        # Compute statistics across all batches for normalization
        print("COMPUTING DATASET STATISTICS\n")

        n_samples = 0
        sum_values = None
        sum_squared = None
        min_values = None
        max_values = None

        batch_count = 0

        for batch in self.loader.load_batches():
            batch_count += 1

            # Drop target column for statistics
            if self.target_column in batch.columns:
                features = batch.drop(columns=[self.target_column])
            else:
                features = batch

            # Select only numeric columns
            numeric_batch = features.select_dtypes(include=[np.number])

            if sum_values is None:
                sum_values = numeric_batch.sum()
                sum_squared = (numeric_batch ** 2).sum()
                min_values = numeric_batch.min()
                max_values = numeric_batch.max()
            else:
                sum_values += numeric_batch.sum()
                sum_squared += (numeric_batch ** 2).sum()
                min_values = np.minimum(min_values, numeric_batch.min())
                max_values = np.maximum(max_values, numeric_batch.max())

            n_samples += len(batch)

            if batch_count % 100 == 0:
                print(f"   Processed {batch_count} batches ({n_samples} rows)...")

        # Compute final statistics
        mean_values = sum_values / n_samples
        variance = (sum_squared / n_samples) - (mean_values ** 2)
        std_values = np.sqrt(variance)

        self.stats = {
            'n_samples': n_samples,
            'mean': mean_values.to_dict(),
            'std': std_values.to_dict(),
            'min': min_values.to_dict(),
            'max': max_values.to_dict()
        }

        print(f"\n   Total samples: {n_samples:,}")
        print(f"   Statistics computed for {len(mean_values)} numeric columns")

        return self.stats

    def preprocess_batch(self, batch: pd.DataFrame, drop_cols: list = None,
                         fill_missing: str = 'mean', normalize: bool = True) -> pd.DataFrame:
        """
        Preprocess a single batch

        Args:
            batch: Input batch
            drop_cols: Columns to drop
            fill_missing: Strategy for missing values ('mean', 'median', 'zero', 'drop')
            normalize: Whether to normalize numeric features

        Returns:
            Preprocessed batch
        """
        # Drop specified columns
        if drop_cols:
            batch = batch.drop(columns=drop_cols, errors='ignore')

        # Separate features and target
        if self.target_column in batch.columns:
            y = batch[self.target_column]
            X = batch.drop(columns=[self.target_column])
        else:
            y = None
            X = batch

        # Handle missing values in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if fill_missing == 'mean':
            for col in numeric_cols:
                if col in self.stats['mean']:
                    X[col] = X[col].fillna(self.stats['mean'][col])
        elif fill_missing == 'median':
            X[col] = X[col].fillna(X[col].median())
        elif fill_missing == 'zero':
            X = X.fillna(0)
        elif fill_missing == 'drop':
            X = X.dropna()
            if y is not None:
                y = y.loc[X.index]

        # Handle categorical columns (simple encoding)
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('missing')
            # Label encoding for simplicity
            X[col] = pd.Categorical(X[col]).codes

        # Normalize numeric features
        if normalize and self.stats:
            for col in numeric_cols:
                if col in self.stats['mean'] and col in self.stats['std']:
                    if self.stats['std'][col] > 0:
                        X[col] = (X[col] - self.stats['mean'][col]) / self.stats['std'][col]

        # Recombine with target
        if y is not None:
            processed = pd.concat([X, y], axis=1)
        else:
            processed = X

        return processed

    def preprocess_and_save(self, output_file: str, drop_cols: list = None,
                            fill_missing: str = 'mean', normalize: bool = True,
                            train_split: float = 0.8):

        # Preprocess all data and save to files
        print("PREPROCESSING AND SAVING DATA\n")

        # Ensure statistics are computed
        if not self.stats:
            print("\n   Computing statistics first...")
            self.compute_statistics()

        train_file = output_file.replace('.csv', '_train.csv')
        test_file = output_file.replace('.csv', '_test.csv')

        first_batch = True
        total_train = 0
        total_test = 0
        batch_count = 0

        for batch in self.loader.load_batches():
            batch_count += 1

            # Preprocess batch
            processed = self.preprocess_batch(batch, drop_cols, fill_missing, normalize)

            # Split into train/test
            n_train = int(len(processed) * train_split)
            train_batch = processed.iloc[:n_train]
            test_batch = processed.iloc[n_train:]

            # Save train
            if len(train_batch) > 0:
                if first_batch:
                    train_batch.to_csv(train_file, index=False, mode='w')
                else:
                    train_batch.to_csv(train_file, index=False, mode='a', header=False)
                total_train += len(train_batch)

            # Save test
            if len(test_batch) > 0:
                if first_batch:
                    test_batch.to_csv(test_file, index=False, mode='w')
                else:
                    test_batch.to_csv(test_file, index=False, mode='a', header=False)
                total_test += len(test_batch)

            first_batch = False

            if batch_count % 100 == 0:
                logger.info(f"   Processed {batch_count} batches (train: {total_train:,}, test: {total_test:,})...")

        logger.info(f"\n   Preprocessing complete!")
        logger.info(f"   Train samples: {total_train:,} → {train_file}")
        logger.info(f"   Test samples: {total_test:,} → {test_file}")

        return train_file, test_file

    def save_preprocessing_config(self, output_dir: str = 'artifacts'):
        # Save preprocessing statistics and config
        Path(output_dir).mkdir(exist_ok=True)

        import json

        # Save statistics
        stats_to_save = {
            'n_samples': self.stats['n_samples'],
            'mean': self.stats['mean'],
            'std': self.stats['std']
        }

        with open(f'{output_dir}/preprocessing_stats.json', 'w') as f:
            json.dump(stats_to_save, f, indent=2)

        logger.info(f"\n   Preprocessing config saved to {output_dir}/")

# Sample use case

if __name__ == "__main__":
    loader = DataLoader(
        data_file = 'data/census-bureau.data',
        columns_file = 'data/census-bureau.columns',
        batch_size = 128
    )

    # Data Exploration
    explorer = DataExplorer(loader)
    insights = explorer.explore()

    # Data visualization
    explorer.visualize()
    
    # Preprocess data
    preprocessor = DataPreprocessor(
    loader=loader,
    target_column=insights['target_column']
    )
    
    # Compute statistics
    preprocessor.compute_statistics()
    
    # Process and save
    train_file, test_file = preprocessor.preprocess_and_save(
    output_file='preprocessed_data.csv',
    drop_cols=insights['missing_cols'] if len(insights['missing_cols']) > 5 else None,
    fill_missing='mean',
    normalize=True,
    train_split=0.8
    )

    # Save
    preprocessor.save_preprocessing_config()
