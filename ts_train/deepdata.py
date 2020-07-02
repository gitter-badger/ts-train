"""
Data transformation for deep learning (:mod:`ts_train.deepdata`)
==========================================

.. currentmodule:: ts_train.deepdata

.. autosummary::
   dataframe_to_dataset
   
.. autofunction:: dataframe_to_dataset

     

"""
import pandas as pd
import tensorflow as tf

def dataframe_to_dataset(df: pd.DataFrame, features: list, target: str):
   """
    Transform DataFrame format to Tensorflow Dataset
    :Parameters:
    
        df: pd.DataFrame
            DataFrame containing multiple columns of features and target
        
    :Returns:
        
        dataset: tf.data.Dataset
            Returns dataset compatible for deeplearning tasks
   """
   dataset = (
       tf.data.Dataset.from_tensor_slices(
           (
               tf.cast(df[features].values, tf.float32),
               tf.cast(df[target].values, tf.int32)
           )
       )
   )
   return dataset
