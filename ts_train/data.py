"""
Data transformation (:mod:`ts_train.data`)
==========================================

.. currentmodule:: ts_train.data

.. autosummary::
   clean_time_series
   interpolate_series
   series2matrix
   create_sequences
   window_generation_for_tsfresh
   transfrom_all_data
   make_features
   reduce_mem_usage
   deduplicate_column_names
   
.. autofunction:: clean_time_series
.. autofunction:: interpolate_series
.. autofunction:: series2matrix
.. autofunction:: create_sequences
.. autofunction:: window_generation_for_tsfresh
.. autofunction:: transfrom_all_data
.. autofunction:: make_features
.. autofunction:: reduce_mem_usage
.. autofunction:: deduplicate_column_names

"""
from typing import Union, List

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d


__docformat__ = 'restructuredtext'
__all__ = ['series2matrix']


def clean_time_series(inSeries: pd.Series) -> pd.Series:
    """
    Remove duplicated based on timestamp index and
    perform linear interpolation for missing timestamp index

    :Parameters:
    
        inSeries: pd.Series
            The time series to be clean from duplicates and fill missing
            by interpolation
        
    :Returns:
        
        return: pd.Series
            Returns clean series
    """
    inSeries.index = pd.to_datetime(inSeries.index)

    mask_duplicated = inSeries.index.duplicated()
    print("Duplicated data points found:", sum(mask_duplicated))
    inSeries = inSeries[~mask_duplicated]

    new_idx = pd.date_range(inSeries.index.min(), inSeries.index.max(), freq="s")
    outSeries = inSeries.reindex(new_idx)
    print("Missing data points found:", sum(outSeries.isna()))
    outSeries = outSeries.interpolate()

    return outSeries


def interpolate_series(time_series: pd.Series, n_points: int, method: str="spline") -> pd.Series:
    """
    Up-sample & Interpolate the pattern to `n_points` number of values.

    :Parameters:

        time_series: pd.Series
            Time series data to model & interpolated  to n_points.

        n_points: int
            Number of points to interpolate for.

    :Returns:
        
        return: pd.Series
            Series with index `n_points` and the interpolated values.
    
    """
    
    if method=="spline":
        spline = CubicSpline(time_series.index, time_series.values)
    else:
        spline = interp1d(time_series.index, time_series.values, kind="nearest")
    
    interpolation_points = np.linspace(0, time_series.index.max(), n_points).round(5)

    return pd.Series(
        data=spline(interpolation_points),
        index=interpolation_points
    )


def series2matrix(in_series: Union[pd.Series, np.ndarray, list], w: int=50, padding: str="valid") -> pd.DataFrame:
    """
    Generate matrix with rolling window from 1D iterator.

    :Parameters:
        
        in_series: pd.Series, np.array or list
            1D iterator

        w: int
            rolling window size

        padding: str
            ["valid", "same"]

    :Returns:
        
        return: pd.DataFrame
            DataFrame of rows as moving windows

    :Examples:
        
        >>> import numpy as np
        >>> import pandas as pd
        >>> ls = [np.random.random() for i in range(10_000)]
        >>> sr = pd.Series(ls) # sr = np.array(ls) # sr = ls
        >>> data_df = series2matrix(sr, w=2526)
        >>> assert data.shape == (7475, 2526)
    
    """
    in_series = in_series.copy()
    in_series.name = None
    
    df = pd.DataFrame(in_series, columns=["t"])
    for i in range(1, w):
        df['t+'+str(i)] = df['t'].shift(-i)

    if padding == "same":
        return df.fillna(0)
    
    return df.dropna()

   
def window_generation_for_tsfresh(series, w=21, id_shift=None):
    """
    Window generation for tsfresh
    
    w = odd windowsize
    window is right aligned inclusive of current position (row)
    
    :Example:
        >>> df = pd.DataFrame()
        >>> df["t"] = np.arange(1, 200)
        >>> sample_windows_df = window_generation_for_tsfresh(df["t"], w=5)
        >>> print(sample_windows_df.shape)
        >>> sample_windows_df[:6]
            sample_id	order	 t
                    2	0      0.0
                        1      1.0
                        2      2.0
                    3	0      1.0
                        1      2.0
                        2      3.0
    """

    org_col = f"t+{(w//2)}"
    df = pd.DataFrame({org_col: series})
    
    if not id_shift:
        id_shift = -(w // 2) - 1
        
    # left-shift
    for i in range(1, (w // 2) + 1):
        df["t+" + str((w // 2) - i)] = df[org_col].shift(i)

    # right-shift
    for i in range(1, (w // 2) + 1):
        df["t+" + str(i + w // 2)] = df[org_col].shift(-i)

    df = df.fillna(0)
    df["sample_id"] = df.index.to_series().shift(periods=id_shift)
    df = df.dropna()
    df["sample_id"] = df["sample_id"].astype("i")

    df = pd.wide_to_long(df, stubnames="t", i=["sample_id"], j="order", sep="+")
    return df.sort_index(0)
   
   
def create_sequences(values: np.array, time_steps: int=32, skip_steps: int=1, ignore_last: int=0):
    """
    Generated training sequences for use in the model.
    
    :Examples:
        >>> x_train = create_sequences(train_feature.values, time_steps=TIME_STEPS, ignore_last=0)
        >>> print("Training input shape: ", x_train.shape)
    """
    output = []
    for i in range(0, len(values) - time_steps, skip_steps):
        output.append(values[i : (i + time_steps - ignore_last)])
        
    return np.stack(output)


def transfrom_all_data(transformer, train: pd.DataFrame, test: pd.DataFrame, feature_list=None):
    """
    Apply transformer to train and test features
    
    :Example:
        >>> logTrans = FunctionTransformer(np.log1p)
        >>> train_trans, test_trans = transfrom_all_data(transformer, train, test, feature_list)
    """
    train_trans = transformer.fit_transform(train[feature_list])
    if len(test):
        test_trans = transformer.transform(test[feature_list])
    else:
        test_trans = None
    
    if type(train_trans) != np.ndarray:
        train_trans = np.array(train_trans)
        if len(test):
            test_trans = np.array(test_trans)
    
    return train_trans, test_trans


def make_features(transformer, train: pd.DataFrame, test: pd.DataFrame, feature_list: List[str], name: str, 
                  normalize: bool=False, scaler=None):
    """
    Add newly generated transformed features to train and test dataframe
    
    :Example:
        >>> scaler = StandardScaler()
        >>> logTrans = FunctionTransformer(np.log1p)
        >>> train_X, val_X = make_features(qTrans, train_X, val_X, feature_list=range(10), name="qTrans", normalize=False, scaler=scaler)
    """
    train, test = train.copy(), test.copy()
    train_trans, test_trans = transfrom_all_data(transformer, train, test, feature_list)
    
    if normalize and scaler is not None:
        train_trans = scaler.fit_transform(train_trans).astype(np.float32)
        test_trans = scaler.transform(test_trans).astype(np.float32)
    
    for i in range(train_trans.shape[1]):
        train['{0}_{1}'.format(name, i)] = train_trans[:, i]
        if len(test):
            test['{0}_{1}'.format(name, i)] = test_trans[:, i]
        
    return train, test

def deduplicate_column_names(df):
    """
    Deduplicate column names by adding suffix f"_{i}" 
    """
    df.columns = [
        x[1]
        if x[1] not in df.columns[: x[0]]
        else f"{x[1]}_{list(df.columns[:x[0]]).count(x[1])}"
        for x in enumerate(df.columns)
    ]
    return df


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by converting data types of numerical columns
    
    :Example:
         >>> df = reduce_mem_usage(df)
             Mem. usage decreased to 124.30 MB (6.0 % reduction)
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduction = (start_mem - end_mem) / start_mem

    msg = (
        f"Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)"
    )
    if verbose:
        print(msg)

    return df
