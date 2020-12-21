"""
Deep Learning utilities (:mod:`ts_train.deeputils`)
================================================================
.. currentmodule:: ts_train.deeputils
.. autosummary::
   plot_lcurve
   add_l1l2_regularizer
   TimedStopping
   
.. autofunction:: plot_lcurve
.. autofunction:: add_l1l2_regularizer
.. autoclass:: TimedStopping
"""
import time
import os
import tempfile
from typing import List 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K


def plot_lcurve(model, metrics: List[str]=["loss"], img_name: str="default.jpg"):
    '''
    Plot Learning Curve
    
    :Parameters:
    
      model: keras.models.Model
         Fitted Model
      
      metrics: List[str]
         Metrics to plot
      
      img_name: str
         Name & Path for image file to save
      
    :Example: 
        >>> model = get_model()
        >>> model.compile()
        >>> model.fit()
        >>> plot_lcurve(model)
    '''
    fig, ax = plt.subplots(len(metrics),1, figsize=(8, 6))
    if len(metrics)==1:
        ax=[ax]
      
    for metric, axi in zip(metrics, ax):
        axi.set_title(metric)
        axi.plot(model.history.history[metric], ".--", label="tr")
        axi.plot(model.history.history[f'val_{metric}'], ".--", label="val")
        axi.legend()
    plt.tight_layout()
    plt.savefig(img_name)
    plt.show()


class TimedStopping(Callback):
    '''
    Stop training when enough time has passed.
    
    :Parameters:
        
        seconds: 
            maximum time before stopping.
        
        safety_factor: 
            stop safety_factor * average_time_per_epoch earlier
        
        verbose: 
            verbosity mode.
    
    :References:
        https://github.com/keras-team/keras/issues/1625#issuecomment-278336908
    '''
    def __init__(self, seconds=None, safety_factor=1, verbose=0):
        super(Callback, self).__init__()

        self.start_time = 0
        self.safety_factor = safety_factor
        self.seconds = seconds
        self.verbose = verbose
        self.time_logs = []

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.start_time
        self.time_logs.append(elapsed_time)

        avg_elapsed_time = float(sum(self.time_logs)) / \
            max(len(self.time_logs), 1)

        print(" ", self.seconds - self.safety_factor * avg_elapsed_time)
        if elapsed_time > self.seconds - self.safety_factor * avg_elapsed_time:
            self.model.stop_training = True
            if self.verbose:
                print('Stopping after %s seconds.' % self.seconds)


class SGDRScheduler(Callback):
    '''
    Cosine annealing learning rate scheduler with periodic restarts.
    :Example:
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    :Parameters:
        
        min_lr: 
            The lower bound of the learning rate range for the experiment.
        
        max_lr: 
            The upper bound of the learning rate range for the experiment.
        
        steps_per_epoch: 
            Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        
        lr_decay: 
            Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        
        cycle_length: 
            Initial number of epochs in a cycle.
        
        mult_factor: 
            Scale epochs_to_restart after each full cycle completion.
    :References:
        Original paper: 
            http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
      

def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    """
    Add L1L2 regularization to the whole model with pre-trained weights.
    - NOTE: This will save and reload the model. Do not call this function inplace but with
    
    :Example:
        >>> model = add_l1l2_regularizer(model, ...)
    
    """
    if not reg_attributes:
        reg_attributes = ['kernel_regularizer', 'bias_regularizer',
                          'beta_regularizer', 'gamma_regularizer']
    if isinstance(reg_attributes, str):
        reg_attributes = [reg_attributes]

    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model
