import pytest
import pandas as pd
import numpy as np
from cleanlab_issue_handler import IssueHandler
from pmlb import fetch_data

# Helper to create a small classification dataset
def make_small_classification_df(n_samples=5, n_features=3):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    df = pd.DataFrame(X, columns=[f'feat{i}' for i in range(n_features)])
    df['target'] = y
    return df

# ------------------------
# Test 1: Remove all issue types
# ------------------------
def test_clean_remove_all_issue_types():
    df = make_small_classification_df(100, 5)
    handler = IssueHandler(df, task="classification")
    original_df, _ = handler.report_issues()
    cleaned_df = handler.clean_selected_issues(
        method="remove",
        label_issues=True,
        outliers=True,
        near_duplicates=True,
        non_iid=True
    )
    assert len(cleaned_df) <= len(original_df)

# ------------------------
# Test 2: Replace labels in classification
# ------------------------
def test_clean_replace_classification():
    df = make_small_classification_df(50, 4)
    handler = IssueHandler(df, task="classification")
    original_df, _ = handler.report_issues()
    cleaned_df = handler.clean_selected_issues(method="replace")
    assert 'target' in cleaned_df.columns
    assert not cleaned_df.equals(original_df)  # Should differ due to replacements

# ------------------------
# Test 3: All flags disabled → no cleaning
# ------------------------
def test_all_flags_disabled():
    df = make_small_classification_df(50, 4)
    handler = IssueHandler(df, task="classification")
    original_df, _ = handler.report_issues()
    cleaned_df = handler.clean_selected_issues(
        method="remove",
        label_issues=False,
        outliers=False,
        near_duplicates=False,
        non_iid=False
    )
    assert cleaned_df.equals(original_df)  # Nothing should be removed

# ------------------------
# Test 4: Missing issue column handling
# ------------------------
def test_partial_issue_columns():
    df = make_small_classification_df(50, 4)
    handler = IssueHandler(df, task="classification")
    handler.report_issues()
    
    # Remove an issue column manually
    if 'is_outlier_issue' in handler.issues.columns:
        handler.issues.drop(columns=['is_outlier_issue'], inplace=True)
    
    try:
        cleaned_df = handler.clean_selected_issues(method="remove")
        assert isinstance(cleaned_df, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"clean_selected_issues failed with partial issue columns: {e}")


# ------------------------
# Test 5: test with PMLB datasets
# ------------------------

def run_classification_test(dataset_name):
    df = fetch_data(dataset_name, return_X_y=False)
    handler = IssueHandler(df, task="classification")
    dataset, issues = handler.report_issues()

    assert isinstance(dataset, pd.DataFrame)
    assert 'target' in dataset.columns
    assert isinstance(issues, pd.DataFrame)
    assert not issues.empty

def run_regression_test(dataset_name):
    df = fetch_data(dataset_name, return_X_y=False)
    handler = IssueHandler(df, task="regression")
    dataset, issues = handler.report_issues()

    assert isinstance(dataset, pd.DataFrame)
    assert 'target' in dataset.columns
    assert 'label_quality' in issues.columns
    assert 'is_label_issue' in issues.columns

@pytest.mark.parametrize("dataset_name", ["breast_cancer", "connect_4"])
def test_classification_pmlb_datasets(dataset_name):
    run_classification_test(dataset_name)

@pytest.mark.parametrize("dataset_name", ["197_cpu_act"])
def test_regression_pmlb_datasets(dataset_name):
    run_regression_test(dataset_name)

