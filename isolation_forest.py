# Install external libraries (uncomment if needed in your environment)
# !pip install ucimlrepo
# !pip install imbalanced-learn

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import randint
from tqdm.notebook import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

# Optionally, suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------
# 1. Download the SECOM Dataset
# -----------------------

# Fetch the dataset from the UCI repository. The dataset is identified by its name 'secom'
data = fetch_ucirepo("secom")

# Extract the original DataFrame from the dataset
X = data.data["original"]

# Optionally, update column names from headers if they are available
headers = data.data.get("headers")
if headers is not None and len(headers) == X.shape[1]:
    X.columns = headers

print("Shape of the SECOM dataset (original):", X.shape)

# Extract the target labels if available
if "targets" in data.data and data.data["targets"] is not None:
    y = pd.Series(data.data["targets"])
    print("Shape of the target labels:", y.shape)
else:
    y = None

# Drop non-numeric columns (such as a date column) before modeling
X_numeric = X.select_dtypes(include=[np.number])
print("Shape after keeping only numeric columns:", X_numeric.shape)

# -----------------------
# 2. Data Preprocessing
# -----------------------

# Impute missing values with the mean on numeric data only
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_numeric)

# Scale the numeric features using StandardScaler for proper model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------
# 3. Apply Isolation Forest for Anomaly Detection
# -----------------------

# Initialize the Isolation Forest.
# The 'contamination' parameter should reflect the expected fraction of anomalies.
isoforest = IsolationForest(contamination=0.05, random_state=42)
isoforest.fit(X_scaled)

# Get the anomaly predictions:
# -1 indicates an anomaly (i.e. a defective wafer)
# 1 indicates a normal observation (nondefective)
predictions = isoforest.predict(X_scaled)

# Append predictions to the original DataFrame for analysis
# (This adds the anomaly predictions even to non-numeric columns.)
X["anomaly"] = predictions

# If true labels are available, evaluate using a classification report.
# Here, anomalies (-1) are mapped to '1' (defective) and normals (1) to '0' (nondefective).
if y is not None:
    pred_labels = (predictions == -1).astype(int)
    print(
        "Classification Report comparing true labels and Isolation Forest predictions:"
    )
    print(classification_report(y, pred_labels))
else:
    print("No true labels available. Proceeding with unsupervised anomaly scores.")

# -----------------------
# 4. Visualize the Anomaly Predictions
# -----------------------

# Plot the distribution of predicted anomalies and normals
sns.countplot(x="anomaly", data=X)
plt.title("Anomaly Distribution by Isolation Forest")
plt.xlabel("Prediction (-1: Anomaly, 1: Normal)")
plt.ylabel("Count")
plt.show()

# Optionally, view a few sample rows with their anomaly predictions
print(X[["anomaly"]].head())
