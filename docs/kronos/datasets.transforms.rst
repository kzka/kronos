Transforms (Data Augmentation)
==============================

Transforms perform data pre-processing and augmentation of video frames.
Specifically, transforms are *consistent* across the time dimension, i.e.
the same transformation (e.g. random crop) is applied to every single
frame in a video sequence.

.. autoclass:: kronos.datasets.transforms.TransformationType
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource

-------------------------------------------------------------------------------

Transforms
----------

.. autoclass:: kronos.datasets.transforms.UnNormalize

.. autoclass:: kronos.datasets.transforms.ColorJitter

.. autoclass:: kronos.datasets.transforms.Augmentor
