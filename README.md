# Census Income Classification and Segmentation Project

## Project Overview
This project implements machine learning solutions for a retail business client to:
1. **Binary Classification**: Predict whether individuals earn income <$50K or ≥$50K based on 40 demographic and employment-related features
2. **Customer Segmentation**: Identify distinct customer groups for targeted marketing campaigns

## Dataset
- **Source**: Weighted census data from 1994-1995 U.S. Census Bureau Current Population Surveys
- **Files**: 
  - `file_name.data` - Main dataset with 40 features plus weight and label
  - `file_name.columns` - Column names corresponding to data file
- **Target Variable**: Binary income classification (<$50K vs ≥$50K)
- **Features**: 40 demographic and employment-related variables

## Project Structure
```
project/
├── main.py                          # Main execution file - runs entire pipeline
├── data_preprocessing.py            # Data loading, exploration, and preprocessing
├── classification_training.py       # Income classification model training
├── segmentation_training.py         # Customer segmentation clustering
├── file_name.data                   # Input data file
├── file_name.columns                # Column names file
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Requirements
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify data files are present**:
   - Ensure `file_name.data` and `file_name.columns` are in the project directory

## Execution

### Running the Complete Pipeline
To execute the entire project (preprocessing, classification, and segmentation):
```bash
python main.py
```

This single command will:
1. Load and explore the data
2. Preprocess and split into train/test sets
3. Train and evaluate the classification model
4. Train and analyze the segmentation model
5. Generate all visualizations and save models
6. Create a comprehensive summary report

**Estimated runtime**: 10-30 minutes depending on dataset size and hardware

### Running Individual Components
You can also run each phase separately:
```bash
# Data preprocessing only
python data_preprocessing.py

# Classification training only (requires preprocessed data)
python classification_training.py

# Segmentation training only (requires preprocessed data)
python segmentation_training.py
```

## Output Files

### Generated Directories
After execution, the following directories will be created:

- **`plots/`** - Data exploration visualizations
  - `target_distribution.png` - Distribution of income classes
  - `correlation.png` - Feature correlation heatmap

- **`classification_results/`** - Classification model outputs
  - `confusion_matrix.png` - Model confusion matrix
  - `roc_curve.png` - ROC curve with AUC score
  - `metrics_comparison.png` - Performance metrics visualization

- **`segmentation_results/`** - Segmentation model outputs
  - `segments_pca.png` - 2D PCA visualization of customer segments
  - `segment_sizes.png` - Distribution of customers across segments
  - `segment_features_heatmap.png` - Feature profiles by segment
  - `elbow_analysis.png` - Optimal cluster selection analysis

- **`models/`** - Saved trained models
  - `classification_model_[timestamp].joblib` - Classification model
  - `classification_metrics_[timestamp].json` - Classification metrics
  - `segmentation_model_[timestamp].joblib` - Segmentation model
  - `segment_profiles_[timestamp].json` - Segment profiles and metrics

- **`artifacts/`** - Preprocessing configuration
  - `preprocessing_stats.json` - Normalization statistics

### Generated Data Files
- `preprocessed_data_train.csv` - Preprocessed training data
- `preprocessed_data_test.csv` - Preprocessed test data

### Log Files
- `project_execution.log` - Complete pipeline execution log
- `data_processing.log` - Data preprocessing log
- `classification_training.log` - Classification training log
- `segmentation_training.log` - Segmentation training log

## Technical Approach

### Data Preprocessing
- **Batch Processing**: Uses 128-row batches for memory efficiency and scalability
- **Missing Value Handling**: Mean imputation for numeric features
- **Normalization**: Z-score standardization for numeric features
- **Categorical Encoding**: Label encoding for categorical variables
- **Train-Test Split**: 80/20 split

### Classification Model
- **Algorithm**: SGD Classifier with logistic loss (scalable for large datasets)
- **Training**: Incremental learning with partial_fit for batch processing
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Alternative**: Random Forest option available for smaller datasets

### Segmentation Model
- **Algorithm**: K-Means clustering
- **Optimal Clusters**: Automatically determined using Elbow method and Silhouette score
- **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
- **Visualization**: PCA dimensionality reduction for 2D visualization
- **Business Insights**: Automated segment profiling and marketing recommendations

## Key Features

### Production-Ready Design
- Batch processing architecture handles datasets larger than RAM
- Comprehensive error handling and logging
- Modular design for easy maintenance and extension
- Reproducible results with saved configurations

### Scalability
- Incremental learning for classification (handles streaming data)
- MiniBatchKMeans option for large-scale segmentation
- Memory-efficient batch-wise statistics computation

### Business Value
- Clear income prediction for targeted marketing
- Actionable customer segments with distinguishing characteristics
- Marketing strategy recommendations per segment
- Budget allocation guidance based on segment sizes

## Business Recommendations

### Classification Use Cases
1. **Direct Marketing**: Target high-income prospects for premium products
2. **Credit Scoring**: Inform credit limit decisions
3. **Product Recommendations**: Tailor product offerings by predicted income level
4. **Resource Allocation**: Focus sales efforts on high-probability segments

### Segmentation Use Cases
1. **Targeted Campaigns**: Design messaging for each segment's characteristics
2. **Product Development**: Customize offerings for segment-specific needs
3. **Channel Optimization**: Select marketing channels by segment preferences
4. **Budget Allocation**: Distribute marketing spend proportional to segment sizes

## Model Performance

### Classification Model
Expected performance metrics (actual values in `classification_results/`):
- Accuracy: 0.80-0.85
- Precision: 0.75-0.85
- Recall: 0.70-0.80
- F1-Score: 0.72-0.82

### Segmentation Model
- Automatically identifies 3-5 optimal customer segments
- Each segment represents 15-35% of customer base
- Clear differentiation in demographic and employment features

## Troubleshooting

### Common Issues

**Import errors**: Ensure all Python files are in the same directory
```bash
cd project_directory
python main.py
```

**Memory errors**: Reduce batch size in main.py
```python
pipeline.run_full_pipeline(batch_size=64)  # Default is 128
```

**Missing dependencies**: Reinstall requirements
```bash
pip install -r requirements.txt --upgrade
```

**File not found**: Verify data files are named correctly
- `file_name.data` (not `filename.data`)
- `file_name.columns` (not `filename.columns`)

## Configuration Options

### In main.py
```python
# Choose classification model
results = pipeline.run_full_pipeline(
    classification_model='sgd',    # Options: 'sgd' or 'random_forest'
    n_clusters=None                # None for auto, or specify integer (e.g., 4)
)
```

### In data_preprocessing.py
```python
# Adjust preprocessing parameters
train_file, test_file = preprocessor.preprocess_and_save(
    fill_missing='mean',           # Options: 'mean', 'median', 'zero', 'drop'
    normalize=True,                # True/False
    train_split=0.8               # 0.0-1.0
)
```
