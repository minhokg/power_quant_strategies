import logging

import numpy as np
import pandas as pd


def remove_nans_iteratively_nparray(
    matrix: np.ndarray,
    date_over_feature_preference_factor: float = 1.1,
) -> np.ndarray:
    """
    Remove rows or columns iteratively with the most NaNs until no NaNs remain.

    :param matrix: 2D numpy array with NaNs
    :param date_over_feature_preference_factor: float, factor to prefer dropping dates over features
    :return: numpy array with NaNs removed
    """
    mask = np.full(matrix.shape, fill_value=False, dtype=bool)

    # iteratively drop worst feature or date until no NaNs remain
    masked_matrix = np.ma.array(matrix, mask=mask)
    while np.isnan(masked_matrix).any() or not masked_matrix.mask.all():
        nan_unmasked = np.isnan(masked_matrix.data) & ~masked_matrix.mask
        n_nans_in_features = np.sum(nan_unmasked, axis=0)
        n_nans_in_dates = np.sum(nan_unmasked, axis=1)

        denom_cols = (~masked_matrix.mask).shape[0] - masked_matrix.mask.sum(axis=0)
        denom_rows = (~masked_matrix.mask).shape[1] - masked_matrix.mask.sum(axis=1)
        denom_cols = np.where(denom_cols == 0, 1, denom_cols)
        denom_rows = np.where(denom_rows == 0, 1, denom_rows)

        fraction_nans_in_features = n_nans_in_features / denom_cols
        fraction_nans_in_dates = n_nans_in_dates / denom_rows

        if fraction_nans_in_features.size == 0 or fraction_nans_in_dates.size == 0:
            # All rows/cols are masked, this can only happen if all dates are fully masked (we should've exited the loop already) or the dataframe is empty
            raise RuntimeError("All rows/cols masked; cannot determine worst index.")

        # identify worst feature and date
        worst_feature_idx = np.argmax(fraction_nans_in_features)
        worst_date_idx = np.argmax(fraction_nans_in_dates)
        feature_drop_threshold = date_over_feature_preference_factor * fraction_nans_in_dates[worst_date_idx]
        if fraction_nans_in_features[worst_feature_idx] >= feature_drop_threshold:
            # drop feature if it has more NaNs than the worst date (considering preference factor)
            masked_matrix.mask[:, worst_feature_idx] = True
        else:
            # drop date otherwise
            masked_matrix.mask[worst_date_idx, :] = True

    if masked_matrix.mask.all():
        raise RuntimeError("All rows/cols masked; cannot determine worst index.")
    # cut away masked rows and columns
    keep_rows = ~(masked_matrix.mask.all(axis=1))
    keep_cols = ~(masked_matrix.mask.all(axis=0))
    compact_matrix = masked_matrix.data[keep_rows][:, keep_cols]
    return compact_matrix


def remove_nans_iteratively_dataframe(
    df: pd.DataFrame,
    date_over_feature_preference_factor: float = 1.1,
) -> pd.DataFrame:
    """
    Remove rows or columns iteratively with the most NaNs until no NaNs remain.

    :param df: DataFrame with NaNs
    :param date_over_feature_preference_factor: float, factor to prefer dropping dates over features
    :return: DataFrame with NaNs removed
    """
    if date_over_feature_preference_factor <= 0.0:
        raise ValueError("date_over_feature_preference_factor must be greater than 0.0")
    df_copy = df.copy()
    logging.debug(f"Starting with shape: {df_copy.shape}")

    # reset datetime index because of potential duplicated execution times
    df_copy.reset_index(inplace=True, names="datetime_index")

    while df_copy.isna().values.any():
        if df_copy.empty:
            raise RuntimeError("Could not determine worst date index. Too many Nans.")
        nans_in_features = np.sum(df_copy.isnull(), axis=0)
        nans_in_dates = np.sum(df_copy.isnull(), axis=1)
        fraction_nans_in_features = nans_in_features / df_copy.shape[0]
        fraction_nans_in_dates = nans_in_dates / df_copy.shape[1]

        # determine largest fraction of nans in features and dates
        worst_feature_idx = fraction_nans_in_features.idxmax()
        worst_date_idx = fraction_nans_in_dates.idxmax()

        # if there are multiple worst indices returned by idxmax, select the first one
        feature_drop_threshold = date_over_feature_preference_factor * fraction_nans_in_dates[worst_date_idx]
        if fraction_nans_in_features[worst_feature_idx] >= feature_drop_threshold:
            # find all features that have a worse or equal fraction of NaNs than the worst date (considering preference factor)
            worst_features_idxs = fraction_nans_in_features[fraction_nans_in_features >= feature_drop_threshold].index
            logging.debug(f"  -> Dropping {len(worst_features_idxs)} features")

            # drop features if they has more NaNs than the worst date (considering preference factor)
            df_copy = df_copy.drop(columns=worst_features_idxs)
        else:
            # find all dates that have a worse or equal fraction of NaNs than the worst feature (considering preference factor)
            worst_dates_idxs = fraction_nans_in_dates[fraction_nans_in_dates >= (fraction_nans_in_features[worst_feature_idx] / date_over_feature_preference_factor)].index
            logging.debug(f"  -> Dropping {len(worst_dates_idxs)} dates")

            # drop dates otherwise
            df_copy = df_copy.drop(index=worst_dates_idxs)
    logging.debug(f"New shape: {df_copy.shape}")

    # restore datetime index
    df_copy.set_index("datetime_index", inplace=True)
    return df_copy
