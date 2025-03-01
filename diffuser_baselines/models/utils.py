from numbers import Number
from typing import Any, Optional, Union
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal


class TemperatureTanh(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        """The hyperbolic tangent with an optional temperature."""
        super().__init__()
        assert temperature != 0.0, "temperature must be nonzero."
        self._T = temperature
        self.tanh = torch.nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.tanh(x / self._T)


class TruncatedNormal(nn.Module):
    """The truncated normal distribution is derived from the normal
    distribution and is bounded above, below, or by both. It is parameterized
    by the mean and variance of the untruncated normal distrubtion. This is
    a custom implementation because it doesn't exist in pytorch.
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Union[float, Tensor],
        smin: float = -np.inf,
        smax: float = np.inf,
        validate_args: Optional[Any] = None,
    ) -> None:
        super().__init__()
        assert smin < smax, "smin must be less than smax"
        assert np.isfinite(smin) and np.isfinite(
            smax
        ), "two-sided truncation is required for now. Set both `smin` and `smax`."
        assert (loc >= smin).all() and (
            loc <= smax
        ).all(), f"loc is out of range ({smin}, {smax}): {loc}"
        if isinstance(scale, Number):
            assert scale >= 0.0, "scale is negative"
        else:
            assert (scale >= 0.0).all(), "scale is negative"

        self._normal = Normal(loc, scale, validate_args=validate_args)
        self._loc = loc
        self._scale = scale
        self._smin = smin
        self._smax = smax
        self._unbounded = self._smin == -np.inf and self._smax == np.inf
        self.A = 1 / (self._scale * np.sqrt(2 * np.pi))
        self.Z = self._normal.cdf(self._smax) - self._normal.cdf(self._smin)
        self.support = constraints.interval(self._smin, self._smax)
        self._init_mean_variance_entropy()

    def _init_mean_variance_entropy(self) -> None:
        """References for entropy:
        https://github.com/olmjo/RcppTN
        https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        standard_normal = Normal(0.0, 1.0)
        standard_normal.pdf = lambda x: (np.e ** (-0.5 * (x ** 2))) / np.sqrt(
            2 * np.pi
        )
        alpha = (self._smin - self._loc) / self._scale
        beta = (self._smax - self._loc) / self._scale

        alpha_pdf = standard_normal.pdf(alpha)
        beta_pdf = standard_normal.pdf(beta)

        alpha_cdf = standard_normal.cdf(alpha)
        beta_cdf = standard_normal.cdf(beta)
        standard_Z = beta_cdf - alpha_cdf

        self._mean = self._loc - self._scale * (
            (beta_pdf - alpha_pdf) / standard_Z
        )

        t1 = (beta * beta_pdf - alpha * alpha_pdf) / standard_Z
        t2 = ((beta_pdf - alpha_pdf) / standard_Z) ** 2
        self._variance = (self._scale ** 2) * (1 - t1 - t2)

        self._entropy = 0.5 * np.log(2 * np.pi * np.e)
        self._entropy += torch.log(self._scale * standard_Z)
        self._entropy += (alpha * alpha_pdf - beta * beta_pdf) / (
            2 * standard_Z
        )

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    def sample(self, resample_limit: int = 10000) -> Tensor:
        if self._unbounded:
            return self._normal.sample()

        samples = self._normal.sample()
        do_resample = (samples < self._smin).logical_or(samples > self._smax)
        num_resamples = 0
        while do_resample.any():
            assert (
                num_resamples < resample_limit
            ), f"Hit resample limit of {resample_limit} for bounds [{self._smin}, {self._smax}]"
            num_resamples += 1

            samples[do_resample] = self._normal.sample()[do_resample]
            do_resample = (samples < self._smin).logical_or(
                samples > self._smax
            )

        return samples

    def log_prob(self, value: Union[float, Tensor]) -> Tensor:
        if self._unbounded:
            return self._normal.log_prob(value)

        msg = "value is out of truncation range and has an undefined log_prob."
        if isinstance(value, Number):
            assert value >= self._smin and value <= self._smax, msg
        else:
            assert (value >= self._smin).all() and (
                value <= self._smax
            ).all(), msg

        normal_prob_density = self.A * np.e ** (
            -0.5 * ((value - self._loc) / self._scale) ** 2
        )
        truncated_prob_density = normal_prob_density / self.Z

        if isinstance(truncated_prob_density, Number):
            return np.log(truncated_prob_density)
        else:
            return truncated_prob_density.log()

    def mode(self):
        return self._loc

    def entropy(self):
        return self._entropy


class DotProductAttention(nn.Module):
    def __init__(self, key_dimension: int) -> None:
        super().__init__()
        self.scale = torch.tensor(1.0 / ((key_dimension) ** 0.5))
        self.softmax = nn.Softmax(dim=2)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot-product attention with an optional mask.
        2X speed improvement over `torch.einsum`.
        Args:
            query: [Batch, Dk]
            key: [Batch, Dk, P]
            value: [Batch, Dv, P]
        Returns:
            tensor of dimension [Batch, Dv]
        """
        energy = torch.bmm(Q.unsqueeze(1), K)
        if mask is not None:
            energy *= mask.unsqueeze(1).float()

        attn = self.softmax(energy * self.scale)
        return torch.bmm(attn, V.permute(0, 2, 1)).squeeze(1)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        d_q_in: int,
        d_k_in: int,
        d_v_in: int,
        d_qk: int,
        d_v: int,
        num_heads: int,
        d_out: int,
        normalize: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        """The residual connection of Vaswani et al is not used here. The
        residual makes sense if self-attention is being used.
        Args:
            d_q_in (int): dimension of the query vector input
            d_k_in (int): dimension of the key vector input
            d_v_in (int): dimension of the value vector input
            d_qk (int): dimension to map queries & keys to prior to attention
            d_v (int): dimension to map values to prior to attention
            num_heads (int): number of attention heads
            d_out (int): output dimension of this module (final linear layer)
        """
        super().__init__()
        self.num_heads = num_heads
        self.normalize = normalize
        self.q_linear = nn.Linear(d_q_in, d_qk * num_heads, bias=False)
        self.k_linear = nn.Linear(d_k_in, d_qk * num_heads, bias=False)
        self.v_linear = nn.Linear(d_v_in, d_v * num_heads, bias=False)

        self.attn = DotProductAttention(d_qk)
        self.final_linear = nn.Linear(d_v * num_heads, d_out, bias=False)

        self.dropout = None
        if dropout_p > 0.0:
            self.dropout = nn.Dropout(dropout_p)

        if self.normalize:
            self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, mask: None = None
    ) -> Tensor:
        """Performs multihead scaled dot product attention for some Q, K, V.
        Args:
            Q: [Batch, d_q_in]
            K: [Batch, d_k_in, P]
            V: [Batch, d_v_in, P]
        """
        assert K.shape[2] == V.shape[2], "keys must be the same size as values"

        Q = self.q_linear(Q)
        K = self.k_linear(K.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        V = self.v_linear(V.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        Q = Q.view(Q.shape[0] * self.num_heads, Q.shape[1] // self.num_heads)
        K = K.view(
            K.shape[0] * self.num_heads,
            K.shape[1] // self.num_heads,
            K.shape[2],
        )
        V = V.view(
            V.shape[0] * self.num_heads,
            V.shape[1] // self.num_heads,
            V.shape[2],
        )

        attended_V = self.attn(Q, K, V, mask=mask)

        attended_V = attended_V.view(
            attended_V.shape[0] // self.num_heads,
            self.num_heads,
            attended_V.shape[1],
        )

        attended_V = attended_V.view(
            attended_V.shape[0], attended_V.shape[1] * attended_V.shape[2]
        )

        out = self.final_linear(attended_V)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.normalize:
            out = self.layer_norm(out)
        return out


class CustomFixedCategorical(torch.distributions.Categorical):
    """Same as the CustomFixedCategorical in hab-lab, but renames log_probs
    to log_prob. All the torch distributions use log_prob.
    """

    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def batched_index_select(
    x: torch.Tensor, dim: int, index: torch.LongTensor
) -> torch.Tensor:
    """A batched index_select where each batch selects different indices.

    Args:
        x: size [B, d0, d1, ..., dn]
        dim: int where 0 <= dim < len(x.size())
        index: size [B, d0, d1, ..., dn]

    Returns:
        torch.Tensor where the selected dimension has been squeezed.

    Example:
        >>> x = torch.randn(2,3,4)
        >>> index = torch.randint(0,3, (2,))
        >>> result = batched_index_select(x, 1, index)  # size: [2, 4]
    """
    views = [x.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(x.shape))
    ]
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(x, dim, index).squeeze(dim)


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class UnMaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label):
        
        self.reduction='none'
        unweighted_loss = super(UnMaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        
        return unweighted_loss
    


def MaskedWeightedLoss(loss_seq, valid_len, inflection_weights):
    
    
    weights = torch.ones_like(inflection_weights) 
    weights = sequence_mask(weights, valid_len)
    
    overall_weights = inflection_weights * weights

    weighted_weights = (loss_seq * overall_weights).sum(dim=1)
    weighted_loss = torch.where(
            valid_len != 0, weighted_weights / valid_len, torch.tensor(0.0, device=loss_seq.device)
        )
    
    return weighted_loss


from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from habitat.utils import profiling_wrapper
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from collections import defaultdict

@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))

from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from diffuser_baselines.models.common.layers import FFWRelativeCrossAttentionModule
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

class NormalLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        compressed_mem = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class MemoryLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self,
        *args,
        **kwargs):

        super().__init__(*args, **kwargs)
        self.memory_layer_intergration_attention = FFWRelativeCrossAttentionModule(4096,2,1)

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        compressed_mem = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # integrating memory
        x_context,_ = self.memory_layer_intergration_attention(query=hidden_states.transpose(0, 1),
            value=compressed_mem.transpose(0, 1),
            query_pos=None,
            value_pos=None,
            diff_ts=None,pad_mask=None)
        x_context = x_context[-1].transpose(0,1)

        hidden_states += 0.2 * x_context

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class MemoryLlamaModel(LlamaModel):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        compressed_mem = None,) -> Union[Tuple, BaseModelOutputWithPast]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        index = 0

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:

                device = hidden_states.device
                props = torch.cuda.get_device_properties(device)
                total_memory = props.total_memory / 1024**2  
                allocated_memory = torch.cuda.memory_allocated(device) / 1024**2  
                reserved_memory = torch.cuda.memory_reserved(device) / 1024**2  
                print("IDX {} Total memory: {:.2f} MB\nMemory allocated: {:.2f} MB\nMemory reserved (cached): {:.2f} MB \n".format(index, total_memory, allocated_memory, reserved_memory))
                index += 1
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    compressed_mem = compressed_mem,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class MemoryLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self,config):

        super().__init__(config)
        self.model = MemoryLlamaModel(config)

    # add memory
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        compressed_mem = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            compressed_mem = compressed_mem,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )