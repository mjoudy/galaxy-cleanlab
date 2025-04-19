import argparse
import numpy as np
import pandas as pd
import pmlb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from temp.cleanlab import Datalab

class CleanlabDataProcessor:
    def __init__(self, dataset_name, model=None):
        self.dataset_name = dataset_name
        self.model = model if model else LogisticRegression(max_iter=1000, random_state=42)
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.merged_df = None
        self.pred_probs = None
        self.lab = None
    
    def load_data(self):
        ds = pmlb.fetch_data(self.dataset_name)
        X = ds.drop('target', axis=1)
        Y = ds['target']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
        df_train = x_train.copy()
        df_train['label'] = y_train
        df_train['split'] = 'train'
        df_test = x_test.copy()
        df_test['label'] = y_test
        df_test['split'] = 'test'
        self.merged_df = pd.concat([df_train, df_test], ignore_index=True)
        self.pred_probs = cross_val_predict(self.model, X, Y, cv=self.cv, method='predict_proba')
        self.lab = Datalab(data=self.merged_df, label_name='label')
    
    def report_issues(self):
        self.lab.find_issues(pred_probs=self.pred_probs)
        return self.lab.report()
    
    def fix_label_issues(self):
        label_issues_df = self.lab.get_issues('label')
        label_issues = label_issues_df[label_issues_df['is_label_issue']].index
        if not label_issues.empty:
            self.merged_df.loc[label_issues, 'label'] = np.argmax(self.pred_probs[label_issues], axis=1)
        return len(label_issues)
    
    def fix_outlier_issues(self):
        outlier_issues_df = self.lab.get_issues('outlier')
        outlier_issues = outlier_issues_df[outlier_issues_df['is_outlier_issue']].index
        if not outlier_issues.empty:
            self.merged_df.drop(outlier_issues, inplace=True)
        return len(outlier_issues)
    
    def get_cleaned_data(self):
        cleaned_train_df = self.merged_df[self.merged_df['split'] == 'train'].drop(columns=['split'])
        cleaned_test_df = self.merged_df[self.merged_df['split'] == 'test'].drop(columns=['split'])
        X_cleaned = cleaned_train_df.drop(columns=['label'])
        y_cleaned = cleaned_train_df['label']
        return X_cleaned, y_cleaned, cleaned_test_df
    
    def evaluate_model(self, X, y):
        pred_probs_cleaned = cross_val_predict(self.model, X, y, cv=self.cv, method='predict_proba')
        oos_preds_cleaned = np.argmax(pred_probs_cleaned, axis=1)
        return accuracy_score(y, oos_preds_cleaned)
    
    def run_cleaning_pipeline(self):
        self.load_data()
        initial_report = self.report_issues()
        fixed_labels = self.fix_label_issues()
        fixed_outliers = self.fix_outlier_issues()
        X_cleaned, y_cleaned, _ = self.get_cleaned_data()
        cleaned_accuracy = self.evaluate_model(X_cleaned, y_cleaned)
        return initial_report, fixed_labels, fixed_outliers, cleaned_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cleanlab Data Processor')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--output_report', type=str, required=True, help='File path to save the issues report')
    parser.add_argument('--output_cleaned_data', type=str, required=True, help='File path to save the cleaned data')
    args = parser.parse_args()

    processor = CleanlabDataProcessor(args.dataset_name)
    initial_report, fixed_labels, fixed_outliers, cleaned_accuracy = processor.run_cleaning_pipeline()

    # Save the report
    with open(args.output_report, 'w') as report_file:
        report_file.write(f"Initial Report:\n{initial_report}\n")
        report_file.write(f"Fixed Label Issues: {fixed_labels}\n")
        report_file.write(f"Fixed Outlier Issues: {fixed_outliers}\n")
        report_file.write(f"Cleaned Accuracy: {cleaned_accuracy:.4f}\n")

    # Save the cleaned data
    X_cleaned, y_cleaned, _ = processor.get_cleaned_data()
    cleaned_data = pd.concat([X_cleaned, y_cleaned], axis=1)
    cleaned_data.to_csv(args.output_cleaned_data, index=False)
