import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from cleanlab import Datalab
from cleanlab.regression.rank import get_label_quality_scores
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

# -------------------
# Base Evaluator Class
# -------------------

class BaseEvaluator:
    def __init__(self, name, dataset, model, task, cv_folds=3):
        self.name = name
        self.dataset = dataset
        self.model = model
        self.task = task
        self.cv_folds = cv_folds
        self.X = dataset.drop('target', axis=1)
        self.y = dataset['target']

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()
        self.train_time = end - start
        self.y_pred = self.model.predict(X_test)
        self.y_test = y_test
        self.X_test = X_test

    def evaluate(self):
        raise NotImplementedError("Must implement in subclass.")

    def log_results(self, path="training_log.csv", cleaned_dataset_size=None, num_issues=None):
        result = pd.DataFrame([{ 
            'dataset': self.name,
            'dataset_size': len(self.dataset),
            'cleaned_dataset_size': cleaned_dataset_size,
            'num_issues': num_issues,
            'task': self.task,
            'model': self.model.__class__.__name__,
            'metric': self.metric,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'train_time': self.train_time
        }])
        result.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)

# -------------------
# Classification Evaluator
# -------------------
class ClassificationEvaluator(BaseEvaluator):
    def evaluate(self):
        self.metric = accuracy_score(self.y_test, self.y_pred)
        scores = cross_val_score(self.model, self.X, self.y, cv=self.cv_folds, scoring='accuracy')
        self.cv_mean, self.cv_std = scores.mean(), scores.std()
        print(f"[Classification] {self.name} - {self.model.__class__.__name__}: Accuracy={self.metric:.4f} | CV={self.cv_mean:.4f}±{self.cv_std:.4f}")

# -------------------
# Regression Evaluator
# -------------------
class RegressionEvaluator(BaseEvaluator):
    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        self.metric = np.sqrt(mse)
        scores = cross_val_score(self.model, self.X, self.y, cv=self.cv_folds, scoring='neg_mean_squared_error')
        self.cv_mean = np.sqrt(-scores.mean())
        self.cv_std = np.sqrt(scores.std())
        print(f"[Regression] {self.name} - {self.model.__class__.__name__}: RMSE={self.metric:.4f} | CV={self.cv_mean:.4f}±{self.cv_std:.4f}")

# -------------------
# Issue Handler
# -------------------

# ================================================================
# ISSUE HANDLING OVERVIEW
# ================================================================
# Current capabilities:
#
# 1. Issue Types Supported (via Cleanlab/Datalab):
#    - is_label_issue           → Incorrect or noisy label
#    - is_outlier_issue         → Statistical outliers
#    - is_near_duplicate_issue  → Redundant or nearly identical samples
#    - is_non_iid_issue         → Violates i.i.d. assumptions
#
# 2. Task Support:
#    - Classification:
#        ✔ All 4 issue types supported via Cleanlab's Datalab
#        ✔ Clean method options: 'remove' and 'replace'
#        ✔ Replacement is done using model's top predicted class (argmax)
#
#    - Regression:
#        ✔ Only label issues supported (detected via residual-based label quality score)
#        ✔ Clean method: 'remove' only
#        ✘ 'replace' method not yet implemented for regression
#
# 3. Cleaning Method Options (via clean_selected_issues method):
#    - method='remove':
#        → Removes all selected issue types from dataset
#    - method='replace':
#        → Replaces label issues with model predictions (classification only)
#
# Ultimate goal:
#    → Extend support for all issue types in both tasks (classification & regression)
#    → Implement label replacement for regression (e.g., smoothing or predictive correction)
#
# Dependencies: Cleanlab (Datalab), scikit-learn, XGBoost (for classification baseline)
# ================================================================


class IssueHandler:
    def __init__(self, dataset, task, n_splits=3, quality_threshold=0.2, knn_k=10):
        self.dataset = dataset
        self.task = task
        self.n_splits = n_splits
        self.quality_threshold = quality_threshold  # threshold for label quality in regression task. labels with quality < threshold are considered issues.
        self.knn_k = knn_k
        self.issues = None
        self.features = None
        self.knn_graph = None
        self.pred_probs = None
        self.issue_summary = None

    def report_issues(self):
        X = self.dataset.drop('target', axis=1)
        y = self.dataset['target']

        # Apparently, cleanlab can have better performance if kkn_graph and features are pre-computed 
        # and passed to it. The longest distance is knn_k +1 and we can study its effect on the performance 
        # of the model in the future.
        nn = NearestNeighbors(n_neighbors=self.knn_k + 1)
        nn.fit(X)
        self.knn_graph = nn.kneighbors(return_distance=False)[:, 1:]



        if self.task == 'classification':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            self.pred_probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

            lab = Datalab(self.dataset, label_name='target')
            lab.find_issues(pred_probs=self.pred_probs, features=self.features, knn_graph=self.knn_graph)
            self.issues = lab.get_issues()  # it gives a table, each column is an issue type for  rows of datapoints.
            self.issue_summary = lab.get_issue_summary()
            print(self.issue_summary)

        elif self.task == 'regression':
            model = LinearRegression()
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            pred_y = cross_val_predict(model, X, y, cv=cv, method='predict')
            scores = get_label_quality_scores(y, pred_y, method='residual')
            self.issues = pd.DataFrame({
                'label_quality': scores,
                'is_label_issue': scores < self.quality_threshold
            })

        return self.dataset.copy(), self.issues.copy()

    def clean_selected_issues(self, method='remove', label_issues=True, outliers=True, near_duplicates=True, non_iid=True):
        if self.issues is None:
            raise RuntimeError("Must run report_issues() before cleaning.")


        # Create a boolean mask marking rows with any kind of selected issue (label, outlier, near-duplicate, non-iid).
        # We loop through each issue type and its corresponding flag (e.g., label_issues=True means we want to include label issues).
        # If the flag is enabled and the issue column exists in self.issues, we update the clean_mask using OR logic.
        # `fillna(False)` ensures that missing values (NaN) are treated as no issue (False).
        # The result: clean_mask is True for rows with at least one selected issue. 
        clean_mask = pd.Series([False]*len(self.dataset))
        for issue_type, use_flag in [
            ('is_label_issue', label_issues),
            ('is_outlier_issue', outliers),
            ('is_near_duplicate_issue', near_duplicates),
            ('is_non_iid_issue', non_iid)
        ]:
            if use_flag and issue_type in self.issues.columns:
                clean_mask |= self.issues[issue_type].fillna(False)

        if method == 'remove':
            return self.dataset[~clean_mask].copy()

        elif method == 'replace' and self.task == 'classification':
            most_likely = np.argmax(self.pred_probs, axis=1)
            fixed = self.dataset.copy()
            to_fix = self.issues['is_label_issue'] & label_issues
            fixed.loc[to_fix, 'target'] = most_likely[to_fix]
            return fixed

        elif method == 'replace' and self.task == 'regression':
            raise NotImplementedError("Replace method not implemented for regression label correction.")

        else:
            raise ValueError("Invalid method or unsupported combination.")
        


