# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Not a contribution.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import ModuleUtilsMixin, apply_chunking_to_forward
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertLayer,
    BertSelfAttention,
    BertSelfOutput,
)

from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
    QuantizedModule,
)
from quantization.base_quantized_model import QuantizedModel
from quantization.quantizers import QMethods
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from quantization.range_estimators import CurrentMinMaxEstimator
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertSelfAttentionWithExtras,
)
from transformers_language.utils import DotDict

# backward-compatibility
HAS_PAST_KEY_ATTR = tuple(map(int, transformers.__version__.split("."))) >= (4, 2, 0)


DEFAULT_QUANT_DICT = {
    # Attention
    "attn_mask_type": "add",
    # Clip `h` tensor
    "k_std": None,
    # LayerNorm
    "layer_norm_ver": "v1",
    "layer_norm_embd": False,
    "layer_norm_res_self_output": False,
    "layer_norm_res_output": False,
    "layer_norm_n_bits_unary": 8,
    "layer_norm_n_bits_binary": 8,
    "layer_norm_n_bits_params": 8,
}


def _make_quant_dict(partial_dict):
    quant_dict = DEFAULT_QUANT_DICT.copy()
    quant_dict.update(partial_dict)
    return DotDict(quant_dict)


class QuantLayerNorm(QuantizedModule):
    def __init__(self, org_module, input_quantizer, **quant_params):
        super().__init__(**quant_params)

        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        self.org_module = org_module
        self.input_quantizer = input_quantizer

        quant_params_ = quant_params.copy()
        quant_params_.update(dict(n_bits_act=self.quant_dict.layer_norm_n_bits_unary))
        self.ln_aq_mu2 = QuantizedActivation(**quant_params_)
        self.ln_aq_S = QuantizedActivation(**quant_params_)
        self.ln_aq_Sigma = QuantizedActivation(**quant_params_)
        self.ln_aq_v = QuantizedActivation(**quant_params_)

        quant_params_ = quant_params.copy()
        quant_params_.update(dict(n_bits_act=self.quant_dict.layer_norm_n_bits_binary))
        self.ln_aq_u = QuantizedActivation(**quant_params_)
        self.ln_aq_w = QuantizedActivation(**quant_params_)
        self.ln_aq_y = QuantizedActivation(**quant_params_)

        self.eps = 1e-12

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)  # mean across last dim
        mu = self.input_quantizer(mu)
        u_q = self.ln_aq_u(x - mu)

        approach = self.quant_dict.layer_norm_ver
        if approach == "v1":
            S = torch.mean(x**2.0, dim=-1, keepdim=True)
            S_q = self.ln_aq_S(S)
            mu2_q = self.ln_aq_mu2(mu * mu)
            Sigma_q = self.ln_aq_Sigma(F.relu(S_q - mu2_q, inplace=True))
        elif approach == "v2":
            Sigma = torch.mean(u_q**2.0, dim=-1, keepdim=True)
            Sigma_q = self.ln_aq_Sigma(Sigma)
        else:
            raise NotImplementedError(f"approach {approach} is not supported")

        v_q = self.ln_aq_v(torch.rsqrt(Sigma_q + self.eps))
        w_q = self.ln_aq_w(u_q * v_q)

        ## quantize gamma, beta
        gamma, beta = self.org_module.weight, self.org_module.bias

        q_gamma = SymmetricUniformQuantizer(
            n_bits=self.quant_dict.layer_norm_n_bits_params, per_channel=False
        )
        r_gamma = CurrentMinMaxEstimator()
        q_gamma.set_quant_range(*r_gamma(gamma))
        gamma_q = q_gamma(gamma)

        q_beta = SymmetricUniformQuantizer(
            n_bits=self.quant_dict.layer_norm_n_bits_params, per_channel=False
        )
        r_beta = (
            CurrentMinMaxEstimator()
        )  # MSE_Estimator(q_beta, opt_method=OptMethod.golden_section)
        q_beta.set_quant_range(*r_beta(beta))
        beta_q = q_beta(beta)

        y_q = self.ln_aq_y(w_q * gamma_q + beta_q)

        return y_q


class QuantizedBertEmbeddings(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()

        quant_params_ = quant_params.copy()
        if "Et" in self.quant_dict:
            from quantization import OptMethod, RangeEstimators

            quant_params_["weight_range_method"] = RangeEstimators.MSE
            quant_params_["weight_range_options"] = dict(opt_method=OptMethod.golden_section)
        self.word_embeddings = quantize_model(org_model.word_embeddings, **quant_params_)

        self.position_embeddings = quantize_model(org_model.position_embeddings, **quant_params)
        self.token_type_embeddings = quantize_model(org_model.token_type_embeddings, **quant_params)

        self.dropout = org_model.dropout

        position_ids = org_model.position_ids
        if position_ids is not None:
            self.register_buffer("position_ids", position_ids)
        else:
            self.position_ids = position_ids

        self.position_embedding_type = getattr(org_model, "position_embedding_type", "absolute")

        # Activation quantizers
        self.sum_input_token_type_embd_act_quantizer = QuantizedActivation(**quant_params)
        self.sum_pos_embd_act_quantizer = QuantizedActivation(**quant_params)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load  # any TensorFlow checkpoint file
        if self.quant_dict.layer_norm_embd:
            self.LayerNorm = QuantLayerNorm(
                org_module=org_model.LayerNorm,
                input_quantizer=self.sum_pos_embd_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.sum_input_token_type_embd_act_quantizer(embeddings)

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            embeddings = self.sum_pos_embd_act_quantizer(embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QuantizedBertSelfAttentionWithExtras(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()

        # copy attributes
        self.num_attention_heads = org_model.num_attention_heads
        self.attention_head_size = org_model.attention_head_size
        self.all_head_size = org_model.all_head_size

        self.position_embedding_type = getattr(org_model, "position_embedding_type", None)
        self.is_decoder = org_model.is_decoder

        # quantized modules
        self.query = quantize_model(org_model.query, **quant_params)
        self.key = quantize_model(org_model.key, **quant_params)
        self.value = quantize_model(org_model.value, **quant_params)
        self.dropout = org_model.dropout

        # Activation quantizers
        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

        # softmax fn
        self.softmax_fn = org_model.softmax_fn

        # attention gating
        self.attn_gate_type = org_model.attn_gate_type
        self.attn_gate_init = org_model.attn_gate_init
        self.attn_gate_mlp = org_model.attn_gate_mlp
        self.attn_gate_mlp2 = org_model.attn_gate_mlp2
        self.attn_gate_linear_all_features = org_model.attn_gate_linear_all_features

        self.alpha = org_model.alpha  # do not quantize for now
        self.gate_fn = org_model.gate_fn
        self.pooling_fn = org_model.pooling_fn

        # self.last_gate_avg_prob = None
        # self.last_gate_all_probs = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores + relative_position_scores_query + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = self.attn_scores_act_quantizer(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # MN: uses our own SM function as specified in the config
        attention_probs = self.softmax_fn(attention_scores, dim=-1)

        # YB: for logging softmax output
        attention_probs = self.attn_probs_act_quantizer(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # YB: for logging softmax output
        # attention_probs = self.attn_probs_after_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # *** Gating ***
        if self.attn_gate_type == AttentionGateType.unconditional_per_head:
            gate = self.gate_fn(self.alpha)  # (H,)
            context_layer *= gate.view(-1, 1, 1)  # (B, H, T, d_head)

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
                x = self.transpose_for_scores(x)  # (B, H, T, d_head)

                alpha = []
                for head_idx in range(self.num_attention_heads):
                    x_head = x[:, head_idx, ...]  # (B, T, d_head)
                    fc_head = self.alpha[head_idx]
                    alpha_head = fc_head(x_head)  # (B, T, 1)
                    if self.attn_gate_type == AttentionGateType.conditional_per_head:
                        alpha_head = self.pooling_fn(alpha_head)  # (B, 1, 1)
                    alpha.append(alpha_head)
                alpha = torch.stack(alpha, dim=1)  # (B, H, *, 1)
                gate = self.gate_fn(alpha)

            context_layer *= gate

            self.last_gate_all_probs = gate  # all gates to see the distributions
            avg_gate = gate.mean(dim=0)
            self.last_gate_avg_prob = avg_gate.view(self.num_attention_heads, -1).mean(dim=1)

        # <end elif>

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.context_act_quantizer(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class QuantizedBertSelfAttention(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()

        # copy attributes
        self.num_attention_heads = org_model.num_attention_heads
        self.attention_head_size = org_model.attention_head_size
        self.all_head_size = org_model.all_head_size

        self.position_embedding_type = getattr(org_model, "position_embedding_type", None)
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            raise NotImplementedError("current branch of computation is not yet supported")

        # quantized modules
        self.query = quantize_model(org_model.query, **quant_params)
        self.key = quantize_model(org_model.key, **quant_params)
        self.value = quantize_model(org_model.value, **quant_params)
        self.dropout = org_model.dropout

        # Activation quantizers
        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            raise NotImplementedError("current branch of computation is not yet supported")

        # NOTE: factor 1/d^0.5 can be absorbed into the previous act. quant. delta
        attention_scores /= math.sqrt(self.attention_head_size)

        attention_scores = self.attn_scores_act_quantizer(attention_scores)

        if attention_mask is not None and self.quant_dict.attn_mask_type == "add":
            # Apply the attention mask is (precomputed for all layers in BertModel forward() fn)
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_probs_act_quantizer(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.context_act_quantizer(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class QuantizedBertSelfOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        # Exact same structure as for BertOutput.
        # Kept in order to be able to disable activation quantizer.
        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        self.dropout = org_model.dropout

        # Activation quantizer
        self.res_act_quantizer = QuantizedActivation(**quant_params)

        # LN
        if self.quant_dict.layer_norm_res_self_output:
            self.LayerNorm = QuantLayerNorm(
                org_module=org_model.LayerNorm,
                input_quantizer=self.res_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.res_act_quantizer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class QuantizedBertOutput(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()

        self.dense = quantize_model(org_model.dense, **quant_params)
        self.dropout = org_model.dropout

        if self.quant_dict.get("y", None) == "log":
            quant_params_ = quant_params.copy()
            quant_params_["act_method"] = QMethods.logarithmic_symmetric
            self.res_act_quantizer = QuantizedActivation(**quant_params_)
        else:
            self.res_act_quantizer = QuantizedActivation(**quant_params)

        # LN
        if self.quant_dict.layer_norm_res_output:
            self.LayerNorm = QuantLayerNorm(
                org_module=org_model.LayerNorm,
                input_quantizer=self.res_act_quantizer.activation_quantizer.quantizer,
                **quant_params,
            )
        else:
            self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.res_act_quantizer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def quantize_intermediate(org_module, **quant_params):
    m_dense = org_module.dense
    m_act = org_module.intermediate_act_fn
    if not isinstance(m_act, nn.Module):
        if m_act == F.gelu:
            m_act = nn.GELU()
        else:
            raise NotImplementedError()
    return quantize_model(nn.Sequential(m_dense, m_act), **quant_params)


class QuantizedBertLayer(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        self.quant_dict = _make_quant_dict(quant_params["quant_dict"])

        super().__init__()

        # copy attributes
        self.chunk_size_feed_forward = org_model.chunk_size_feed_forward
        self.seq_len_dim = org_model.seq_len_dim
        self.is_decoder = org_model.is_decoder
        self.add_cross_attention = org_model.add_cross_attention

        # quantized components
        attention_specials = {
            BertSelfAttention: QuantizedBertSelfAttention,
            BertSelfAttentionWithExtras: QuantizedBertSelfAttentionWithExtras,
            BertSelfOutput: QuantizedBertSelfOutput,
        }
        self.attention = quantize_model(
            org_model.attention, specials=attention_specials, **quant_params
        )
        if self.add_cross_attention:
            self.crossattention = quantize_model(
                org_model.crossattention, specials=attention_specials, **quant_params
            )
        self.intermediate = quantize_intermediate(org_model.intermediate, **quant_params)
        self.output = QuantizedBertOutput(org_model.output, **quant_params)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attn_args = (hidden_states, attention_mask, head_mask)
        attn_kw = dict(output_attentions=output_attentions)
        if HAS_PAST_KEY_ATTR:
            attn_kw["past_key_value"] = past_key_value

        self_attention_outputs = self.attention(*attn_args, **attn_kw)

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        assert self.chunk_size_feed_forward == 0  # below call is a no-op in that case
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class QuantizedBertPooler(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.dense_act = quantize_model(
            nn.Sequential(org_model.dense, org_model.activation), **quant_params
        )

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        # _save_act_tensor(self, first_token_tensor, 'r', layerwise=False)

        pooled_output = self.dense_act(first_token_tensor)
        # _save_act_tensor(self, pooled_output, 'f', layerwise=False)

        # if TB_OLD_LOGS:
        #     _tb_advance_global_step(self)
        return pooled_output


class QuantizedBertModel(QuantizedModel, ModuleUtilsMixin):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.config = org_model.config

        self.embeddings = QuantizedBertEmbeddings(org_model.embeddings, **quant_params)
        self.encoder = quantize_model(
            org_model.encoder, specials={BertLayer: QuantizedBertLayer}, **quant_params
        )
        self.pooler = (
            QuantizedBertPooler(org_model.pooler, **quant_params)
            if org_model.pooler is not None
            else None
        )

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


#
# ** BERT for Masked Language Modeling **
#


class QuantizedBertPredictionHeadTransform(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        if org_model.transform_act_fn == F.gelu:
            transform_act_fn = nn.GELU()
        else:
            raise ValueError(
                f'transform activation fn "{org_model.transform_act_fn}" ' f"is not supported"
            )

        self.dense_act = quantize_model(
            nn.Sequential(org_model.dense, transform_act_fn), **quant_params
        )
        # NOTE: assume naive LN for now
        self.LayerNorm = quantize_model(org_model.LayerNorm, **quant_params)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        # output is already quantized
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class QuantizedBertLMPredictionHead(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.transform = QuantizedBertPredictionHeadTransform(org_model.transform, **quant_params)
        self.decoder = quantize_model(org_model.decoder, **quant_params)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class QuantizedBertForMaskedLM(QuantizedModel):
    def __init__(self, org_model, quant_setup=None, **quant_params):
        super().__init__()

        self.config = org_model.config

        self.bert = QuantizedBertModel(org_model=org_model.bert, **quant_params)
        # self.cls = QuantizedBertOnlyMLMHead(org_model.cls, **quant_params)
        self.cls = org_model.cls

        # NOTE: use FP_logits setup by default
        # self.cls.predictions.decoder.activation_quantizer = FP32Acts()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
