"""
Data transformation (:mod:`ts_train.data`)
==========================================

.. currentmodule:: ts_train.data

.. autosummary::
   clean_time_series
   interpolate_series
   series2matrix
   create_sequences
   transfrom_all_data
   make_features
   
.. autofunction:: clean_time_series
.. autofunction:: interpolate_series
.. autofunction:: series2matrix
.. autofunction:: create_sequences
.. autofunction:: transfrom_all_data
.. autofunction:: make_features

"""
from typing import Union, List

import pandas as pd
import numpy as np
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
