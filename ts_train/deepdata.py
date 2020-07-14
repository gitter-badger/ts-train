"""
Data transformation for deep learning (:mod:`ts_train.deepdata`)
================================================================

.. currentmodule:: ts_train.deepdata

.. autosummary::
   dataframe_to_dataset
   
.. autofunction:: dataframe_to_dataset

"""
import pandas as pd
import tensorflow as tf

def dataframe_to_dataset(df: pd.DataFrame, features: list, target: str):
    """
    Transform DataFrame format to Tensorflow Dataset.

    :Parameters:

        df: pd.DataFrame
            DataFrame containing multiple columns of features and target

        features: list
            List of columns containing feature data

        target: str
            Name of target column

    :Returns:

        dataset: tf.data.Dataset
            Returns dataset compatible for deeplearning tasks

    :Example:
        >>> training_df = pd.DataFrame(
                data={
                    'feature1': np.random.rand(10),
                    'feature2': np.random.rand(10),
                    'feature3': np.random.rand(10),
                    'target': np.random.randint(0, 3, 10)
                    }
            )
        >>> features = ['feature1', 'feature2', 'feature3']
        >>> training_dataset = dataframe_to_dataset(training_df, features, 'target')
        >>> for features_tensor, target_tensor in training_dataset:
                print(f'features:{features_tensor} target:{target_tensor}')

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
