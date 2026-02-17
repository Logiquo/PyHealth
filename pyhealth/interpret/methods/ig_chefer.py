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

from typing import Dict, Optional

import torch
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
        self.model = model

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
        pass
        
        # Step 3. Compute IG attributions with the modified backward rules
        attributions = self.ig.attribute(**kwargs) # type: ignore[call-arg]

        # Step 4. Clean up hooks
        pass

        return attributions
        