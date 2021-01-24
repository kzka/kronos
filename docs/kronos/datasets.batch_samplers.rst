Batch Samplers
==============

Batch samplers are iterators that work in conjunction with :code:`VideoDataset` objects.
They take as input a batch size and the directory tree of a dataset and generate an
iterable of indices specifying which videos to load in each batch for an entire epoch.
Once the iterable is exhausted, a new permutation of the indices is generated.

Base Video Batch Sampler
------------------------

.. autoclass:: kronos.datasets.batch_samplers.VideoBatchSampler
  :members:
  :show-inheritance:
  :member-order: groupwise

-------------------------------------------------------------------------------

Batch Samplers
--------------

.. autoclass:: kronos.datasets.batch_samplers.SameClassBatchSampler

.. autoclass:: kronos.datasets.batch_samplers.SameClassBatchSamplerDownstream
