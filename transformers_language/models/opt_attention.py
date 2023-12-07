# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from functools import partial
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers_language.models.bert_attention import AttentionGateType, logit
from transformers_language.models.softmax import clipped_softmax


class OPTAttentionWithExtras(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        ## new
        softmax_fn=torch.nn.functional.softmax,
        alpha=None,
        max_seq_length=None,
        ssm_eps=None,
        tau=None,
        skip_attn=False,
        attn_gate_type=AttentionGateType.none,
        attn_gate_init=None,
        attn_gate_mlp=False,
        attn_gate_mlp2=False,
        attn_gate_linear_all_features=False,
        fine_tuning=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # YB: capture the input and output of the softmax
        self.attn_scores = nn.Identity()  # before attention mask
        self.attn_probs_before_dropout = nn.Identity()
        self.attn_probs_after_dropout = nn.Identity()

        self.alpha = alpha
        self.max_seq_length = max_seq_length
        self.ssm_eps = ssm_eps
        self.tau = tau

        # define softmax function
        if self.alpha is not None:
            assert self.max_seq_length is not None
            gamma = -self.alpha / self.max_seq_length
            self.softmax_fn = partial(clipped_softmax, gamma=gamma, eta=1.0)
        else:
            self.softmax_fn = softmax_fn

        self.skip_attn = skip_attn

        # attention gating
        self.last_gate_avg_prob = None
        self.last_gate_all_probs = None

        self.attn_gate_type = attn_gate_type
        self.attn_gate_init = attn_gate_init
        self.attn_gate_mlp = attn_gate_mlp
        self.attn_gate_mlp2 = attn_gate_mlp2
        self.attn_gate_linear_all_features = attn_gate_linear_all_features

        self.alpha = None
        self.ssm_eps = ssm_eps
        self.gate_fn = torch.sigmoid
        self.pooling_fn = partial(torch.mean, dim=1, keepdims=True)

        self.fine_tuning = fine_tuning

        # gate scaling factor
        self.gate_scaling_factor = 1.0
        if self.fine_tuning and self.attn_gate_init is not None:
            self.gate_scaling_factor = 1.0 / self.attn_gate_init

        # define gate
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            init_alpha = torch.zeros(size=(self.num_heads,))
            self.alpha = nn.Parameter(init_alpha, requires_grad=True)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            if self.attn_gate_linear_all_features:
                self.alpha = nn.Linear(self.embed_dim, self.num_heads, bias=True)

            else:  # separate predictors for each head
                module_list = []
                for _ in range(self.num_heads):
                    if self.attn_gate_mlp:
                        fc = nn.Sequential(
                            nn.Linear(self.head_dim, self.head_dim // 4, bias=True),
                            nn.ReLU(),
                            nn.Linear(self.head_dim // 4, 1, bias=True),
                        )
                    elif self.attn_gate_mlp2:
                        fc = nn.Sequential(
                            nn.Linear(self.head_dim, self.head_dim, bias=True),
                            nn.ReLU(),
                            nn.Linear(self.head_dim, 1, bias=True),
                        )
                    else:
                        fc = nn.Linear(self.head_dim, 1, bias=True)

                        if self.attn_gate_init is not None:
                            init_bias = logit(self.attn_gate_init)
                            torch.nn.init.constant_(fc.bias, init_bias)

                        if self.fine_tuning:
                            # init to a very small values
                            torch.nn.init.normal_(fc.weight, mean=0.0, std=0.001)

                    module_list.append(fc)
                self.alpha = nn.ModuleList(module_list)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # YB: for logging softmax input
        attn_weights = self.attn_scores(attn_weights)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = self.softmax_fn(attn_weights, dim=-1, dtype=torch.float32).to(
                torch.float16
            )
        else:
            attn_weights = self.softmax_fn(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # YB: for logging softmax output
        attn_weights = self.attn_probs_before_dropout(attn_weights)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # YB: for logging softmax output
        attn_probs = self.attn_probs_after_dropout(attn_probs)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output - (B, H, T, d_head)

        #
        # *** Gating ***
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            gate = self.gate_fn(self.alpha)  # (H,)
            attn_output *= gate.view(-1, 1, 1)  # (B, H, T, d_head)

            self.last_gate_avg_prob = gate.view(-1)

        elif self.attn_gate_type in (
            AttentionGateType.conditional_per_head,
            AttentionGateType.conditional_per_token,
        ):
            x = hidden_states  # (B, T, d_model)

            if self.attn_gate_linear_all_features:  # assume per_token
                alpha = self.alpha(x)  # (B, T, H)
                gate = self.gate_fn(alpha)
                gate = gate.permute(0, 2, 1).contiguous()  # (B, H, T)
                gate = gate.unsqueeze(3)  # (B, H, T, 1)

            else:
                # x = self.transpose_for_scores(x)  # (B, H, T, d_head)
                x = self._shape(x, -1, bsz)  # (B, H, T, d_head)

                alpha = []
                for head_idx in range(self.num_heads):
                    x_head = x[:, head_idx, ...]  # (B, T, d_head)
                    fc_head = self.alpha[head_idx]
                    alpha_head = fc_head(x_head)  # (B, T, 1)
                    if self.attn_gate_type == AttentionGateType.conditional_per_head:
                        alpha_head = self.pooling_fn(alpha_head)  # (B, 1, 1)
                    alpha.append(alpha_head)
                alpha = torch.stack(alpha, dim=1)  # (B, H, *, 1)
                gate = self.gate_fn(alpha)

            attn_output *= gate * self.gate_scaling_factor

            self.last_gate_all_probs = gate  # all gates to see the distributions
            avg_gate = gate.mean(dim=0)
            self.last_gate_avg_prob = avg_gate.view(self.num_heads, -1).mean(dim=1)

        #
        ## <end elif>

        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
