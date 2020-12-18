"""
Deep Learning utilities (:mod:`ts_train.deeputils`)
================================================================
.. currentmodule:: ts_train.deeputils
.. autosummary::
   plot_lcurve
   
.. autofunction:: plot_lcurve
"""
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

def plot_lcurve(model, metrics=["loss"], name="default.jpg"):
    '''Plot Learning Curve
    
    Usage: 
        model = get_model()
        model.compile()
        model.fit()
        plot_lcurve(model)
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
    plt.savefig(name)
    plt.show()



class TimedStopping(Callback):
    '''Stop training when enough time has passed.
    # Arguments
        seconds: maximum time before stopping.
        safety_factor: stop safety_factor * average_time_per_epoch earlier
        verbose: verbosity mode.
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
