"""
Data transformation (:mod:`ts_train.data`)
==========================================

.. currentmodule:: ts_train.data

.. autosummary::
   clean_time_series
   interpolate_series
   series2matrix
   
.. autofunction:: clean_time_series
.. autofunction:: interpolate_series
.. autofunction:: series2matrix

     

"""
from typing import Union, List

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


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


def interpolate_series(time_series: pd.Series, n_points: int) -> pd.Series:
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
    spline = CubicSpline(time_series.index, time_series.values)

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


