pyhealth.models.SequenceEmbeddingModel
======================================

Embeddings for the packed records of ``sequence_v2``
(:class:`~pyhealth.processors.SequenceProcessorV2`) fields. Each level's
records get one embedding per token namespace and a projection of their
values, concatenated; the levels can be kept apart (with their offsets and
times) or flattened into one sequence of innermost records per sample.
:class:`~pyhealth.models.MultimodalRNN2` builds on it.

.. autoclass:: pyhealth.models.SequenceEmbeddingModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.LevelEmbedding
    :members:

.. autoclass:: pyhealth.models.PretrainedEmbedding
    :members:
