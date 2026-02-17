"""Integrated Gradients with Chefer weighted on the integral path
for attention.

This module combines Integrated Gradients (IG) as the core attribution
method with Chefer's attention relevance scores as the weighting function
for the attention contributions along the IG path.  The result is a path-integral
attribution that accounts for the dynamic importance of attention heads
at each interpolation step.

References:
    * Sundararajan et al., "Axiomatic Attribution for Deep Networks",
      ICML 2017. https://arxiv.org/abs/1703.01365
    * Chefer et al., "Generic Attention-Model Explainability for Interpreting
      Transformer Models", NeurIPS 2021. https://arxiv.org/abs/2010.12498
"""

from __future__ import annotations

from typing import Dict, Optional, cast

import torch
from torch import nn
import torch.nn.functional as F

from pyhealth.models import BaseModel

from .base_interpreter import BaseInterpreter
from pyhealth.interpret.api import Interpretable, CheferInterpretable
from .chefer import CheferRelevance
from .integrated_gradients import IntegratedGradients

class CheferWeightedIntegratedGradient(BaseInterpreter):
    """Integrated Gradients with Chefer attention weights.

    This interpreter computes the Chefer relevance scores 
    for each attention. Then we use these relevance scores to weight 
    the attention contributions at each step of the IG path integral, 
    resulting in attributions that reflect both the input gradients 
    and the dynamic importance of attention heads.

    Args:
        model: Trained PyHealth model that implements
            ``forward_from_embedding()`` and ``get_embedding_model()``.
        temperature: Softmax temperature used for the TSG backward rule.
            Values > 1 flatten the softmax Jacobian, counteracting
            gradient suppression.  ``2.0`` is the paper's recommended
            default.
        steps: Default number of Riemann-sum interpolation steps.  Can
            be overridden per call in :meth:`attribute`.
    """

    def __init__(
        self,
        model: BaseModel,
        temperature: float = 1.0,
        steps: int = 50,
    ):
        super().__init__(model)
        
        self.chefer = CheferRelevance(model)
        self.ig = IntegratedGradients(model, steps=steps)
        self.model = cast(CheferInterpretable, model)

        self.temperature = max(float(temperature), 1.0)
        self.steps = steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attribute(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        
        # Step 1. Compute Chefer weights for attention layers
        self.chefer.relevance = {}  # reset so we only capture this call's R
        self.chefer.set_relevance_hooks(True)
        _ = self.chefer.attribute(**kwargs) # type: ignore[call-arg]
        R = self.chefer.relevance
        self.chefer.set_relevance_hooks(False)

        # Step 2. Swap in Chefer-weighted backward rule for attention layers
        old_modules = self.model.get_attention_modules()
        modules: dict[str, list[nn.Module]] = {
            key: [
                CheferWeightedMHA(r, attention, temperature=self.temperature)
                for r, attention 
                in zip(R[key], attentions)
            ]
            for key, attentions
            in old_modules.items()
        }
        self.model.set_attention_modules(modules)
        
        # Step 3. Compute IG attributions with the modified backward rules
        attributions = self.ig.attribute(**kwargs) # type: ignore[call-arg]

        # Step 4. Clean up hooks
        self.model.set_attention_modules(old_modules)

        return attributions

class CheferWeightedMHA(nn.Module):
    """Drop-in replacement for MultiHeadedAttention.
    
    Uses precomputed Chefer relevance matrix instead of softmax(QK^T/sqrt(d)).
    Gradients flow through V projection and output projection normally.
    Q and K projections are skipped entirely.
    """
    
    def __init__(self, weight: torch.Tensor, mha: nn.Module, temperature: float = 1.0):
        """
        Args:
            weight: [batch, heads, seq_len, seq_len] 
                    Precomputed, softmax-normalized Chefer relevance.
                    Treated as a fixed constant (no grad).
            mha: Original MultiHeadedAttention module.
                 We reuse its V projection and output projection.
            temperature: Optional scaling factor for the Chefer weights.
                         Values > 1 flatten the distribution, counteracting
                         gradient suppression.  Default is 1.0 (no scaling).
        """
        super().__init__()
        from pyhealth.models.transformer import MultiHeadedAttention
        
        if not isinstance(mha, MultiHeadedAttention):
            raise ValueError("Unsupported MultiHeadedAttention module.")
        
        self.weight = weight.detach()  # ensure it's treated as a fixed constant
        self.temperature = temperature

        # Copy from MHA
        self.V = mha.linear_layers[2]  # V projection
        self.d_k = mha.d_k
        self.h = mha.h
        self.output_linear = mha.output_linear

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        register_hook: bool = False,
    ) -> torch.Tensor:
        """Run multi-head attention with optional gradient capture.

        Args:
            query: Query tensor ``[batch, len_q, hidden]`` or similar.
            key: Key tensor aligned with ``query``.
            value: Value tensor aligned with ``query``.
            mask: Optional boolean mask ``[batch, len_q, len_k]``.
            register_hook: True to attach a backward hook saving gradients.

        Returns:
            torch.Tensor: Attention mixed representation ``[batch, len_q, hidden]``.
        """
        batch_size = value.size(0)

        # Step 1: V projection only (Q, K are ignored)
        v = self.V(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # v: [batch, heads, seq_len, d_k]

        # Step 2: Apply mask to Chefer weight, then softmax
        w = self.weight  # [batch, heads, seq_len, seq_len] (raw, pre-softmax)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq, seq] for head broadcast
            w = w.masked_fill(mask == 0, -1e9)
        w = F.softmax(w / self.temperature, dim=-1)

        x = torch.matmul(w, v)

        # Step 3: Concat heads and output projection (same as original MHA)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k
        )
        return self.output_linear(x)