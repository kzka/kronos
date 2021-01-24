Tensorizers
===========

Tensorizers encapsulate the logic for ingesting dict-like data structures
used by datasets in :mod:`kronos.datasets.datasets` and converting their
keys to PyTorch tensors. These dicts are returned by the :code:`__getitem__`
method of the datasets and as such contain data for a *single* video only.
The keys of these dicts must be one of the values of the types in :code:`SequenceType`.
This is because for each sequence type, we have a corresponding tensorizer tailored to converting
it to a PyTorch tensor.

.. autoclass:: kronos.datasets.tensorizers.SequenceType
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource

.. autoclass:: kronos.datasets.tensorizers.ToTensor
    :exclude-members: __init__

-------------------------------------------------------------------------------

Tensorizers
-----------

.. autoclass:: kronos.datasets.tensorizers.Tensorizer
    :exclude-members: __init__

.. autoclass:: kronos.datasets.tensorizers.IdentityTensorizer
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: kronos.datasets.tensorizers.LongTensorizer
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: kronos.datasets.tensorizers.FramesTensorizer
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: kronos.datasets.tensorizers.ActorTypeTensorizer
    :show-inheritance:
    :exclude-members: __init__
