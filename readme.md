# Next level deep learning from scratch
This repo includes code for my <b><i>Next level deep learning from scratch</b></i>.
## Requirements
- Python 3.x
- Numpy
- matplotlib
- CuPy (optional)
## Sample code
```python
# import deep learning class
from DeepLearning import NN, Layers, Utils, Losses, Optimizers, Regularizers, Metrics
import numpy as np
import cupy as cp

# read training (100 samples) and validation data

# define model hyperparameters
clf = NN(Losses.softmax_cross_entropy,
         Optimizers.RMSProp,
         regularizer=Regularizers.l1,
         reg_lambda=0.01)
clf.add(Layers.Dense(128, input_dim=100))
clf.add(Layers.Dense(96, activation='lrelu', batch_norm=(0.99, 0.001, 1e-5)))
clf.add(Layers.Dense(64, activation='lrelu', batch_norm=(0.99, 0.001, 1e-5)))
clf.add(Layers.Dense(10, activation='softmax'))

result = clf.fit(x_train,
                 y_train,
                 learning_rate=0.01,
                 batch_size=128,
                 epochs=epochs,
                 print_freq=1,
                 gamma=0.0,
                 decay=0.9,
                 validation=(x_val, y_val))
```
