kronos.evaluators
=================

One a model has been pre-trained with any of the available self-supervised techniques,
it can be evaluated on a number of downstream fine-grained temporal understanding tasks.

.. automodule:: kronos.evaluators
    :no-members:

Base Evaluators
---------------

.. autoclass:: kronos.evaluators.base.Evaluator
    :members:
    :member-order: groupwise

-------------------------------------------------------------------------------

Downstream Evaluators
---------------------

.. autoclass:: kronos.evaluators.CycleConsistency

.. autoclass:: kronos.evaluators.KendallsTau

.. autoclass:: kronos.evaluators.PhaseAlignmentTopK

.. autoclass:: kronos.evaluators.NearestNeighbourVisualizer
