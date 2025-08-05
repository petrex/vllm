# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common attention backend interface.

This module defines :class:`AttentionBackend`, a minimal interface that
all attention backends must implement. The interface focuses on the
forward pass and KV cache management so that out-of-tree backends can be
integrated easily.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch

from vllm.attention.backends.abstract import (  # type: ignore
    AttentionBackend as _LegacyAttentionBackend,
    AttentionLayer,
    AttentionMetadata,
)


class AttentionBackend(_LegacyAttentionBackend):
    """Base interface for attention backends.

    Backends inheriting from this class are required to implement the
    ``forward`` method in addition to the KV cache utility functions
    defined on :class:`~vllm.attention.backends.abstract.AttentionBackend`.
    """

    @staticmethod  # pragma: no cover - interface definition
    @abstractmethod
    def forward(
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the attention kernel.

        Implementations may update ``kv_cache`` in-place and must return the
        attention output tensor. ``output`` can be provided to reuse an
        existing buffer when ``accept_output_buffer`` is set to ``True``.
        """
        raise NotImplementedError
