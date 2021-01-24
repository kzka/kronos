kronos.factories
================

.. autoclass:: kronos.factories.Factory
    :exclude-members: __init__

------------------------------------------------------------------------------

Dataloading-related Factories
-----------------------------

.. autoclass:: kronos.factories.PreTrainingDatasetFactory
    :members: from_config
    :exclude-members: __init__

.. autoclass:: kronos.factories.DownstreamDatasetFactory
    :members: from_config
    :exclude-members: __init__

.. autoclass:: kronos.factories.BatchSamplerFactory
    :members: from_config
    :exclude-members: __init__

------------------------------------------------------------------------------

Modeling-related Factories
--------------------------

.. autoclass:: kronos.factories.ModelFactory
    :members: from_config
    :exclude-members: __init__

------------------------------------------------------------------------------

Training and evaluating-related Factories
-----------------------------------------

.. autoclass:: kronos.factories.TrainerFactory
    :members: from_config
    :exclude-members: __init__

.. autoclass:: kronos.factories.EvaluatorFactory
    :members: from_config
    :exclude-members: __init__

------------------------------------------------------------------------------

Optimization-related Factories
------------------------------

.. autoclass:: kronos.factories.OptimizerFactory
    :members: from_config
    :exclude-members: __init__
