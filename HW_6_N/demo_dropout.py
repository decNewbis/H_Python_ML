# Inverted Dropout

import numpy as np

keep_prob = 0.8
activations = np.random.normal(0.5, 0.5, size=(3, 4))
dropout = np.random.rand(activations.shape[0], activations.shape[1]) < keep_prob
print(activations)
print(dropout)

### Training stage
# `activations /= keep_prob` is to increase `activations` values
# to keep z (`z = w*activations + b`) on the same lavel
activations = np.multiply(activations, dropout)
print(activations)
activations /= keep_prob
print(activations)


### Inference stage
# At inference stage dropout is not used
# -- Option 1:
# run inference with dropout many times and average the results
# -- Option 2:
# DO NOTHING!!! We have already scaled activations and trained NN,
# so there are no need to make another one scaling!

