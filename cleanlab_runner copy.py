import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from cleanlab import Datalab
from cleanlab.regression.rank import get_label_quality_scores

from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import pmlb



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

    def log_results(self, path="training_log.csv"):
        result = pd.DataFrame([{ 
            'dataset': self.name,
            'dataset_size': len(self.dataset),
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
    def __init__(self, dataset, task, method='remove', n_splits=3, quality_threshold=0.2):
        self.dataset = dataset
        self.task = task
        self.method = method
        self.n_splits = n_splits
        self.quality_threshold = quality_threshold

    def clean(self):
        X = self.dataset.drop('target', axis=1)
        y = self.dataset['target']

        if self.task == 'classification':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
            lab = Datalab(self.dataset, label_name='target')
            lab.find_issues(pred_probs=probs)
            issues = lab.get_issues()
            mask = (issues.get('is_outlier_issue', pd.Series([False]*len(y))) |
                    issues.get('is_label_issue', pd.Series([False]*len(y))))
            if self.method == 'remove':
                return self.dataset[~mask].copy(), issues
            elif self.method == 'replace':
                most_likely = np.argmax(probs, axis=1)
                fixed = self.dataset.copy()
                fixed.loc[issues['is_label_issue'], 'target'] = most_likely[issues['is_label_issue']]
                return fixed, issues

        elif self.task == 'regression':
            model = LinearRegression()
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            pred_y = cross_val_predict(model, X, y, cv=cv, method='predict')
            scores = get_label_quality_scores(y, pred_y, method='residual')
            issues = pd.DataFrame({
                'label_quality': scores,
                'is_label_issue': scores < self.quality_threshold
            })
            if self.method == 'remove':
                return self.dataset[~issues['is_label_issue']].copy(), issues
            elif self.method == 'replace':
                fixed = self.dataset.copy()
                fixed.loc[issues['is_label_issue'], 'target'] = pred_y[issues['is_label_issue']]
                return fixed, issues

        else:
            raise ValueError("Task must be 'classification' or 'regression'")
