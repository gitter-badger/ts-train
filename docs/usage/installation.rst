Installation
############

Using pip (recommended)
=======================

1. Just in case: in a terminal, check the **consistency** of the outputs of the commands ``which python``, ``python --version``, ``which pip`` and ``pip --version``. 

2. In a terminal

  .. code-block:: bash

    pip install ts-train
    
    # alternative
    pip install git+https://github.com/minesh1291/ts-train.git
    
From source using git
=====================

1. Clone the ts-train repo at a location of your choice (denoted here as ``/path/to``)

  .. code-block:: console

    git clone https://github.com/minesh1291/ts-train.git /path/to/ts-train
    cd /path/to/ts-train
    python setup.py install
