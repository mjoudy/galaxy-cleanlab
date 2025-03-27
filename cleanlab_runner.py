import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
class IssueHandler:
    def __init__(self, dataset, task, n_splits=3, quality_threshold=0.2, knn_k=10):
        self.dataset = dataset
        self.task = task
        self.n_splits = n_splits
        self.quality_threshold = quality_threshold
        self.knn_k = knn_k
        self.issues = None
        self.features = None
        self.knn_graph = None
        self.pred_probs = None
        self.issue_summary = None

    def report_issues(self):
        X = self.dataset.drop('target', axis=1)
        y = self.dataset['target']

        # Preprocess features
        X_proc = pd.get_dummies(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_proc)
        pca = PCA(n_components=min(20, X_scaled.shape[1]))
        self.features = pca.fit_transform(X_scaled)

        # Compute knn_graph
        nn = NearestNeighbors(n_neighbors=self.knn_k + 1)
        nn.fit(self.features)
        self.knn_graph = nn.kneighbors(return_distance=False)[:, 1:]

        if self.task == 'classification':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            self.pred_probs = cross_val_predict(model, X_proc, y, cv=cv, method='predict_proba')

            lab = Datalab(self.dataset, label_name='target')
            lab.find_issues(pred_probs=self.pred_probs, features=self.features, knn_graph=self.knn_graph)
            self.issues = lab.get_issues()
            self.issue_summary = lab.get_issue_summary()
            print(self.issue_summary)

        elif self.task == 'regression':
            model = LinearRegression()
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            pred_y = cross_val_predict(model, X_proc, y, cv=cv, method='predict')
            scores = get_label_quality_scores(y, pred_y, method='residual')
            self.issues = pd.DataFrame({
                'label_quality': scores,
                'is_label_issue': scores < self.quality_threshold
            })

        return self.dataset.copy(), self.issues.copy()

    def clean_selected_issues(self, method='remove', label_issues=True, outliers=True, near_duplicates=True, non_iid=True):
        if self.issues is None:
            raise RuntimeError("Must run report_issues() before cleaning.")

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