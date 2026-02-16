"""Attention Rollout for transformer interpretability.

This module implements the Attention Rollout method from Abnar & Zuidema
(2020) for explaining transformer-family model predictions.  It relies on
the :class:`~pyhealth.interpret.api.CheferInterpretable` interface — any
model that implements that interface is automatically supported.

Attention Rollout does **not** require a backward pass.  It uses only the
raw attention maps captured during the forward pass, making it a fast and
simple gradient-free attribution method.

The algorithm multiplies attention matrices across layers to track how
information flows from input tokens to the final representation, accounting
for residual connections by blending each attention matrix with the identity.

Paper:
    Abnar, Samira, and Willem Zuidema.
    "Quantifying Attention Flow in Transformers."
    Proceedings of the 58th Annual Meeting of the Association for
    Computational Linguistics (ACL), 2020.
"""

from typing import Dict, Optional

import torch

from pyhealth.interpret.api import CheferInterpretable
from pyhealth.models.base_model import BaseModel
from .base_interpreter import BaseInterpreter


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _avg_heads(attn: torch.Tensor) -> torch.Tensor:
    """Average attention weights across heads.

    Args:
        attn: Attention weights of shape ``[batch, heads, seq, seq]``
            or ``[batch, seq, seq]`` (already head-averaged).

    Returns:
        Head-averaged attention of shape ``[batch, seq, seq]``.
    """
    if attn.dim() == 4:
        return attn.mean(dim=1)
    return attn


def _add_residual(attn: torch.Tensor, residual_weight: float = 0.5) -> torch.Tensor:
    """Add identity (residual connection) to attention matrix.

    Transformer residual connections mean that information can also flow
    directly from input to output, bypassing attention.  We model this
    by blending the attention matrix with the identity:

        A_hat = (1 - w) * A + w * I

    where ``w`` is ``residual_weight``.  The common setting ``w = 0.5``
    means equal weighting of attended and residual paths.

    Args:
        attn: Attention matrix ``[batch, seq, seq]``.
        residual_weight: Weight for the identity (residual) path.
            Must be in ``[0, 1]``.  Default ``0.5`` (equal mix).

    Returns:
        Residual-augmented attention ``[batch, seq, seq]``.
    """
    I = torch.eye(attn.size(-1), device=attn.device, dtype=attn.dtype)
    I = I.unsqueeze(0).expand_as(attn)
    return (1.0 - residual_weight) * attn + residual_weight * I


def _normalize_rows(attn: torch.Tensor) -> torch.Tensor:
    """Row-normalize attention so each row sums to 1.

    Args:
        attn: Attention matrix ``[batch, seq, seq]``.

    Returns:
        Row-normalized attention ``[batch, seq, seq]``.
    """
    row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return attn / row_sums


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

class AttentionRollout(BaseInterpreter):
    """Attention Rollout for transformer interpretability.

    This interpreter works with **any** model that implements the
    :class:`~pyhealth.interpret.api.CheferInterpretable` interface, which
    currently includes:

    * :class:`~pyhealth.models.Transformer`
    * :class:`~pyhealth.models.StageAttentionNet`

    The algorithm (from Abnar & Zuidema, 2020):

    1. Enable attention hooks via ``model.set_attention_hooks(True)``.
    2. Forward pass — capture attention maps (no backward needed).
    3. For each attention layer:
       a. Average attention across heads.
       b. Add identity matrix for residual connections:
          ``A_hat = (1 - w) * A + w * I``.
       c. Optionally row-normalize ``A_hat``.
       d. Accumulate: ``R = A_hat @ R``.
    4. Reduce ``R`` to per-token vectors via
       ``model.get_relevance_tensor()``.

    Steps 1 and 4 are delegated to the model through the
    ``CheferInterpretable`` interface, making this class fully
    model-agnostic.

    Args:
        model (BaseModel): A trained PyHealth model that implements
            :class:`~pyhealth.interpret.api.CheferInterpretable`.
        residual_weight (float): Weight for the identity (residual) path
            when combining with attention.  Must be in ``[0, 1]``.
            Default ``0.5`` (equal mix of attention and residual).
        normalize (bool): Whether to row-normalize the augmented
            attention matrices before multiplication.  Default ``True``.
        discard_ratio (float): Fraction of lowest attention values to
            zero out per layer before rollout.  Default ``0.0`` (keep
            all values).  Setting this to e.g. ``0.1`` can reduce noise.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import AttentionRollout
        >>>
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C", "A06A"],
        ...         "procedures": ["P01", "P02"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v1",
        ...         "conditions": ["A05B"],
        ...         "procedures": ["P01"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ehr_example",
        ... )
        >>> model = Transformer(dataset=dataset)
        >>> # ... train the model ...
        >>>
        >>> interpreter = AttentionRollout(model)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>>
        >>> # Compute attention rollout attributions
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns dict: {"conditions": tensor, "procedures": tensor}
        >>> print(attributions["conditions"].shape)  # [batch, num_tokens]
        >>>
        >>> # With different residual weight (more emphasis on attention)
        >>> interpreter = AttentionRollout(model, residual_weight=0.3)
        >>> attributions = interpreter.attribute(**batch)
    """

    def __init__(
        self,
        model: BaseModel,
        residual_weight: float = 0.5,
        normalize: bool = True,
        discard_ratio: float = 0.0,
    ):
        super().__init__(model)
        if not isinstance(model, CheferInterpretable):
            raise ValueError(
                "Model must implement the CheferInterpretable interface. "
                "Currently supported: Transformer, StageAttentionNet."
            )
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                f"residual_weight must be in [0, 1], got {residual_weight}"
            )
        if not 0.0 <= discard_ratio < 1.0:
            raise ValueError(
                f"discard_ratio must be in [0, 1), got {discard_ratio}"
            )
        self.model = model
        self.residual_weight = residual_weight
        self.normalize = normalize
        self.discard_ratio = discard_ratio

    def attribute(
        self,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute attention rollout attribution scores for each input token.

        Unlike Chefer, this method only requires a forward pass — no
        backward pass is needed since it does not use gradients.

        Args:
            **data: Input data from dataloader batch containing feature
                keys and label key.

        Returns:
            Dict[str, torch.Tensor]: Dictionary keyed by feature keys,
                where each tensor has shape ``[batch, seq_len]`` with
                per-token attribution scores.
        """
        # --- 1. Forward with attention hooks enabled ---
        self.model.set_attention_hooks(True)
        try:
            with torch.no_grad():
                logits = self.model(**data)["logit"]
        finally:
            self.model.set_attention_hooks(False)

        # --- 2. Retrieve attention maps (ignore gradients) ---
        attention_layers = self.model.get_attention_layers()

        batch_size = logits.shape[0]
        device = logits.device

        # --- 3. Rollout: propagate attention across layers ---
        R_dict: dict[str, torch.Tensor] = {}
        for key, layers in attention_layers.items():
            num_tokens = layers[0][0].shape[-1]
            R = (
                torch.eye(num_tokens, device=device, dtype=logits.dtype)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            for attn_map, _attn_grad in layers:
                # Average across heads (no gradient weighting)
                attn = _avg_heads(attn_map.detach())

                # Optionally discard lowest attention values
                if self.discard_ratio > 0.0:
                    flat = attn.view(attn.size(0), -1)
                    k = int(flat.size(-1) * self.discard_ratio)
                    if k > 0:
                        threshold = flat.kthvalue(k, dim=-1).values
                        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
                        attn = attn.masked_fill(attn < threshold, 0.0)

                # Add residual connection (identity)
                attn = _add_residual(attn, self.residual_weight)

                # Row-normalize
                if self.normalize:
                    attn = _normalize_rows(attn)

                # Accumulate rollout
                R = torch.matmul(attn, R)

            R_dict[key] = R

        # --- 4. Reduce R matrices to per-token vectors ---
        attributions = self.model.get_relevance_tensor(R_dict, **data)

        # --- 5. Expand to match raw input shapes (nested sequences) ---
        return self._map_to_input_shapes(attributions, data)

    # ------------------------------------------------------------------
    # Shape mapping (shared logic with CheferRelevance)
    # ------------------------------------------------------------------

    def _map_to_input_shapes(
        self,
        attributions: Dict[str, torch.Tensor],
        data: dict,
    ) -> Dict[str, torch.Tensor]:
        """Expand attributions to match raw input value shapes.

        For nested sequences the attention operates on a pooled
        (visit-level) sequence, but downstream consumers (e.g. ablation
        metrics) expect attributions to match the raw input value shape.
        Per-visit relevance scores are replicated across all codes
        within each visit.

        Args:
            attributions: Per-feature attribution tensors returned by
                ``model.get_relevance_tensor()``.
            data: Original ``**data`` kwargs from the dataloader batch.

        Returns:
            Attributions expanded to raw input value shapes where needed.
        """
        result: Dict[str, torch.Tensor] = {}
        for key, attr in attributions.items():
            feature = data.get(key)
            if feature is not None:
                if isinstance(feature, torch.Tensor):
                    val = feature
                else:
                    schema = self.model.dataset.input_processors[key].schema()
                    val = (
                        feature[schema.index("value")]
                        if "value" in schema
                        else None
                    )
                if val is not None and val.dim() > attr.dim():
                    for _ in range(val.dim() - attr.dim()):
                        attr = attr.unsqueeze(-1)
                    attr = attr.expand_as(val)
            result[key] = attr
        return result
