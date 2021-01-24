kronos.utils.checkpoint
=======================

:code:`Checkpoint` serializes any class implementing
a :code:`state_dict` method (e.g. a model and optimizer). It is used by
:code:`CheckpointManager` to manage different
model checkpoints, providing convenience functions such as only
keeping a maximum of `X` checkpoints at any given time or loading the most
recent checkpoint for resuming a interrupted training runs.

.. code-block:: python

    from kronos.utils.checkpoint import CheckpointManager, Checkpoint

    # create checkpoint manager
    checkpoint_manager = CheckpointManager(
        Checkpoint(model, optimizer),
        checkpoint_dir,
        device,
    )

    # fetch last iteration number
    global_step = checkpoint_manager.restore_or_initialize()

-------------------------------------------------------------------------------

.. autoclass:: kronos.utils.checkpoint.Checkpoint
    :members:
    :member-order: groupwise

.. autoclass:: kronos.utils.checkpoint.CheckpointManager
    :members:
    :member-order: groupwise
