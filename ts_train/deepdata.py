"""
Data transformation for deep learning (:mod:`ts_train.deepdata`)
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
import pandas as pd


def dataframe_to_dataset(df: pd.DataFrame):
    
    return