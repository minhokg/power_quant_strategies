import numpy as np
import pandas as pd
import pytest

from power_quant_strategies.utils.helper.remove_nans import remove_nans_iteratively_dataframe, remove_nans_iteratively_nparray

test_mat_1 = np.array([[1, 2, np.nan, np.nan], [3, 4, np.nan, np.nan], [6, 8, np.nan, 5], [np.nan, np.nan, 9, np.nan]])
test_mat_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_mat_3 = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])

test_mat_1_result = np.array([[1, 2], [3, 4], [6, 8]])
test_mat_2_result = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_mat_3_result = np.array([]).reshape(0, 0)
df_1 = pd.DataFrame(test_mat_1)
df_2 = pd.DataFrame(test_mat_2)
df_3 = pd.DataFrame(test_mat_3)

test_mat_1 = remove_nans_iteratively_nparray(test_mat_1)
test_mat_2 = remove_nans_iteratively_nparray(test_mat_2)
test_mat_3 = remove_nans_iteratively_nparray(test_mat_3)


def test_remove_nans_iteratively_nparray():
    assert np.array_equal(test_mat_1.data, test_mat_1_result)
    assert np.array_equal(test_mat_2.data, test_mat_2_result)
    assert np.array_equal(test_mat_3.data, test_mat_3_result)


def test_remove_nans_iteratively_dataframe():
    df_1_result = pd.DataFrame(test_mat_1_result)
    df_2_result = pd.DataFrame(test_mat_2_result)

    df_1_cleaned = remove_nans_iteratively_dataframe(df_1)
    df_2_cleaned = remove_nans_iteratively_dataframe(df_2)
    df_3_cleaned = remove_nans_iteratively_dataframe(df_3)

    # _row_id keeps the original row position of every kept row (used to align targets)
    assert df_1_cleaned["_row_id"].tolist() == [0, 1, 2]
    assert df_2_cleaned["_row_id"].tolist() == [0, 1, 2]

    assert np.isclose(df_1_result, df_1_cleaned.drop(columns="_row_id")).all().all()
    assert np.isclose(df_2_result, df_2_cleaned.drop(columns="_row_id")).all().all()
    assert df_3_cleaned.drop(columns="_row_id").columns.empty


def test_remove_nans_iteratively_dataframe_rejects_non_positive_factor():
    with pytest.raises(ValueError):
        remove_nans_iteratively_dataframe(df_1, date_over_feature_preference_factor=0.0)


def test_remove_nans_iteratively_nparray_rejects_non_positive_factor():
    with pytest.raises(ValueError):
        remove_nans_iteratively_nparray(np.array([[1.0]]), date_over_feature_preference_factor=0.0)
