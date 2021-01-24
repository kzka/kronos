Frame Samplers
==============

Frame samplers are used by :code:`VideoDataset` objects to specify
which frames in a video to sample. They take as input a video directory
and a number of frames to sample, and return a list of indices specifying
which filepaths in the file tree to sample.

Base Frame Samplers
-------------------

.. autoclass:: kronos.datasets.frame_samplers.FrameSampler
  :members:
  :member-order: groupwise

.. autoclass:: kronos.datasets.frame_samplers.SingleVideoFrameSampler
  :members:
  :member-order: groupwise

.. autoclass:: kronos.datasets.frame_samplers.MultiVideoFrameSampler
  :members:
  :member-order: groupwise

-------------------------------------------------------------------------------

Frame Samplers
--------------

.. autoclass:: kronos.datasets.frame_samplers.StridedSampler
    :show-inheritance:

.. autoclass:: kronos.datasets.frame_samplers.AllSampler
    :show-inheritance:

.. autoclass:: kronos.datasets.frame_samplers.OffsetUniformSampler
    :show-inheritance:

.. autoclass:: kronos.datasets.frame_samplers.WindowSampler
    :show-inheritance:
