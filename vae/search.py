import numpy as np
import subprocess

import hyperopt
from hyperopt import hp

space = {
    'kld-weight': hp.loguniform('kld-weight', 0.1, 2.0),
    'latent-size': hp.quniform('latent-size', 10, 400, 1),
    'batch-size': hp.quniform('batch-size', 4, 6, 1),
    'learning-rate': hp.loguniform('learning-rate', np.log(1e-4),
                                   np.log(0.001)),
    'model': hp.choice('model', ['deep'])
}

for _ in range(10):
    sample = hyperopt.pyll.stochastic.sample(space)

    sample['latent-size'] = int(sample['latent-size'])
    sample['batch-size'] = int(np.power(2, sample['batch-size']))
    sample['epochs'] = 50

    arg_command = ' '.join(
        [f'--{key}={value}' for key, value in sample.items()])
    command = f'floyd run --gpu --data ibebrett/datasets/deepsheep/1:/deepsheep --env=pytorch-1.0 "python main.py /deepsheep {arg_command}"'
    print(command)
    #subprocess.call(command, shell=True)
