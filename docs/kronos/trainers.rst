kronos.trainers
===============

Trainers encapsulate all the logic necessary for training a model on the training
set and running it in `eval` mode on the validation set.

Base Trainers
---------------

.. autoclass:: kronos.trainers.base.Trainer
    :members:
    :member-order: bysource

-------------------------------------------------------------------------------

Trainers
--------

.. autoclass:: kronos.trainers.tcc.TCCTrainer
    :exclude-members: __init__

.. autoclass:: kronos.trainers.sal.SALTrainer
    :exclude-members: __init__

.. autoclass:: kronos.trainers.tcn.SVTCNTrainer
    :exclude-members: __init__
