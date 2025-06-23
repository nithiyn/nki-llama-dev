# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    _gather_along_first_dim,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.utils import cpu_mode
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
from neuronxcc.nki.language import nc
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


# Get the modules_to_not_convert from the neuron configs
def get_modules_to_not_convert(neuron_config: NeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def get_updated_configs(config: InferenceConfig):
    """
    Generate a list of configurations for each hidden layer in a Llama model.

    This function creates a list of InferenceConfig objects, one for each layer. It
    modifies the configurations for certain layers based on which modules should not
    be converted to quantized format. The function uses get_modules_to_not_convert()
    to determine which modules should not be converted.

    Args:
    config (InferenceConfig): The inference configuration for the model.

    Returns:
    list[InferenceConfig]: A list of InferenceConfig objects, one for each layer in the model.
                           Each config may be either the original config or a modified version
                           with "quantized_mlp_kernel_enabled" as False for that specific layer.
    """
    updated_configs = []
    modules_to_not_convert = get_modules_to_not_convert(config.neuron_config)
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for i in range(config.num_hidden_layers):
        # If any of the MLP modules for this layer are in modules_to_not_convert
        module_pattern = f"layers.{i}.mlp"
        if any(module_pattern in module for module in modules_to_not_convert):
            non_quant_config = copy.deepcopy(config)
            non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
            non_quant_config.neuron_config.activation_quantization_type = None
            non_quant_config.neuron_config.quantize_clamp_bound = float("inf")
            updated_configs.append(non_quant_config)
        else:
            updated_configs.append(config)
    return updated_configs


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def _helper_concat_and_delete_qkv(llama_state_dict, layer_num, attr):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        llama_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    llama_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del llama_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del llama_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(llama_state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(llama_state_dict, l, "scale")

    gc.collect()

    return llama_state_dict


class WeightGatheredColumnParallel(ColumnParallelLinear):
    """
    A specialized column-parallel linear layer that implements weight gathering optimization
    for efficient processing of long sequences in transformer models during eagle speculation.

    This layer provides two forward paths:
    1. Standard column-parallel forward (inherited from parent)
    2. Weight-gathered forward for long sequences
    """
    def forward_wg(self, input: torch, weight_gather: bool = False):
        """
        Performs the forward pass with optional weight gathering optimization.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len/TP, 2*hidden_size)
            weight_gather (bool): Whether to use weight gathering optimization.
                                Typically True for sequences >= 32K

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - If skip_bias_add is False: Output tensor of shape (batch_size, seq_len, hidden_size)
                - If skip_bias_add is True: Tuple of (output tensor, bias)
        """
        if weight_gather:
            weight = _gather_along_first_dim(self.weight, process_group=self.tensor_parallel_group)
            output = self._forward_impl(
                input=input,
                weight=weight,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                process_group=self.tensor_parallel_group
            )

            output = gather_from_sequence_parallel_region(
                output,
                self.sequence_dimension,
                process_group=self.tensor_parallel_group,
            )
            if self.skip_bias_add:
                return output, self.bias

            output = (output + self.bias) if self.bias is not None else output
            return output
        else:
            return self.forward(input)


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.fused_rmsnorm_skip_gamma = self.config.neuron_config.fused_rmsnorm_skip_gamma
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        self.quantize_clamp_bound = self.neuron_config.quantize_clamp_bound
        self.logical_nc_config = self.neuron_config.logical_nc_config
        self.activation_quantization_type = self.neuron_config.activation_quantization_type
        mlp_bias = getattr(config, "mlp_bias", False)

        if self.neuron_config.quantized_mlp_kernel_enabled and self.quantize_clamp_bound == float(
            "inf"
        ):
            logging.warning(
                "quantize_clamp_bound is not specified in NeuronConfig. We will use the default value of 1200 for llama models in quantized kernels."
            )
            self.quantize_clamp_bound = 1200.0
        if parallel_state.model_parallel_is_initialized():
            if self.neuron_config.quantized_mlp_kernel_enabled:
                # # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=False,
                sequence_dimension=None,
                tensor_model_parallel_group=get_tp_group(config),
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=config.neuron_config.torch_dtype,
                pad=True,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                tensor_model_parallel_group=get_tp_group(config),
                reduce_dtype=config.neuron_config.rpl_reduce_dtype,
            )
            if self.mlp_kernel_enabled:
                if self.neuron_config.quantized_mlp_kernel_enabled:
                    setattr(
                        self.gate_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                    setattr(
                        self.up_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                    setattr(
                        self.down_proj,
                        "post_create_quantized_module_hook",
                        preprocess_quantized_linear_layer,
                    )
                else:
                    # Transpose the weights to the layout expected by kernels
                    self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_quantized_mlp(self, x, rmsnorm, residual, adapter_ids):
        grid = (nc(self.logical_nc_config),)
        fused_residual = residual is not None
        fused_rmsnorm = rmsnorm is not None
        logger.debug(
            f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_nc_config={self.logical_nc_config}"
        )

        # Can't do residual add in the kernel if SP is enabled
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "Quantized MLP cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(quant_mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(quant_mlp_isa_kernel)

        if fused_rmsnorm:
            ln_w = rmsnorm.weight.unsqueeze(0)
        else:
            ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)

        # Handle SP RMSnorm
        x_orig_dtype = x.dtype
        if self.sequence_parallel_enabled:
            # This RMSNormQuant kernel will do quantization inside, so we pass the
            # clamp_bound for clipping.
            # If we don't use this kernel, the MLP kernel below will do the
            # quantization, so we also pass clamp_bound to that kernel.
            if self.rmsnorm_quantize_kernel_enabled:
                logger.debug(
                    "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
                )
                _rmsnorm_quant_fwd_call = nki_jit()(rmsnorm_quant_isa_kernel)
                quant_rmsnorm_out = torch.zeros(
                    size=(
                        x.shape[0],  # batch size
                        x.shape[1],  # sequence length
                        x.shape[2] + 4,  # hidden size + 4 bytes for packing fp32 scale
                    ),
                    dtype=torch.int8,
                    device=x.device,
                )
                clamp_bound = self.quantize_clamp_bound
                _rmsnorm_quant_fwd_call[grid](
                    x, ln_w, clamp_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
                )
                x = gather_from_sequence_parallel_region(
                    quant_rmsnorm_out,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                )

            else:
                logger.debug(
                    "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                )
                x = gather_from_sequence_parallel_region(
                    x, self.sequence_dimension, process_group=get_tp_group(self.config)
                )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x_orig_dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        gate_w = self.gate_proj.weight.data
        gate_w_scale = self.gate_proj.scale
        up_w = self.up_proj.weight.data
        up_w_scale = self.up_proj.scale
        down_w = self.down_proj.weight.data
        down_w_scale = self.down_proj.scale
        clamp_bound = self.quantize_clamp_bound

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                clamp_bound,
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                clamp_bound,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        fused_rmsnorm = rmsnorm is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, skip_gamma={self.fused_rmsnorm_skip_gamma}, logical_nc_config={self.logical_nc_config}"
        )

        # Choose which kernel to call
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        if fused_rmsnorm:
            ln_w = rmsnorm.weight.unsqueeze(0)
        else:
            ln_w = torch.zeros(size=(1, self.hidden_size), dtype=x.dtype, device=x.device)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        grid = (nc(self.logical_nc_config),)

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                kernel_name="MLP",
                fused_rmsnorm=fused_rmsnorm,
                skip_gamma=self.fused_rmsnorm_skip_gamma,
                eps=self.rms_norm_eps,
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                kernel_name="MLP",
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                skip_gamma=self.fused_rmsnorm_skip_gamma,
                eps=self.rms_norm_eps,
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, adapter_ids=None):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        gate_proj_output = (
            self.gate_proj(x)
            if not is_lora_module(self.gate_proj)
            else self.gate_proj(x, adapter_ids)
        )

        up_proj_output = (
            self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        )

        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.down_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        logger.debug(f"MLP output shape {output.shape}")
        return output

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel
        If rmsnorm is passed in, will fuse the rmsnorm into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """

        if self.mlp_kernel_enabled:
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, rmsnorm, residual, adapter_ids=adapter_ids
                )
            # MLP kernel
            return self._kernel_enabled_mlp(x, rmsnorm, residual, adapter_ids=adapter_ids)
        else:
            # No kernel
            assert rmsnorm is None and residual is None
            return (self._native_mlp(x, adapter_ids=adapter_ids), None)


@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(tensor_model_parallel_group=tensor_model_parallel_group)

        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype
        self.is_medusa = config.neuron_config.is_medusa
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.bias = getattr(config, "attention_bias", False)
        self.rpl_reduce_dtype = config.neuron_config.rpl_reduce_dtype
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rms_norm_eps = config.rms_norm_eps
        self.attn_tkg_builtin_kernel_enabled = self.neuron_config.attn_tkg_builtin_kernel_enabled

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = self.config.neuron_config.tp_degree
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        logger.debug(
            f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        )

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(self.config)
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)

        if self.attn_tkg_builtin_kernel_enabled:
            self.inv_freqs = self.rotary_emb.get_inv_freqs().unsqueeze(1)


# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
        inv_freq = 1.0 / (self.base ** (freq_indices / self.dim))

        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / self.factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = self.get_inv_freqs(x.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )

        self.mlp = NeuronLlamaMLP(config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = config.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.qkv_kernel_fuse_residual_add = config.neuron_config.qkv_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.is_prefill_stage = config.neuron_config.is_prefill_stage
        self.config = config

        if self.is_prefill_stage and self.config.neuron_config.is_mlp_quantized():
            # for CTE, quantized MLP kernel does not support fused rmsnorm
            self.mlp_kernel_fused_rmsnorm = False
        else:
            self.mlp_kernel_fused_rmsnorm = not self.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,  # residual from previous layer used by QKV
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]], Optional[torch.FloatTensor], Optional[torch.FloatTensor], Optional[torch.FloatTensor]]:
        entry_hidden_states = hidden_states
        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # produced another residual used by MLP
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            rotary_position_ids=rotary_position_ids,
            residual=residual,
            **kwargs,
        )

        if attn_output.residual is None:
            residual = entry_hidden_states  # input to attention
        else:
            # residual will only be returned by attn/qkv if fuse add qkv kernel is enabled
            assert self.qkv_kernel_fuse_residual_add, \
                "residual add before qkv should be computed in the previous layer, \
                 unless qkv_kernel_fuse_residual_add is specified"
            assert (
                not self.sequence_parallel_enabled
            ), "qkv_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            assert (
                self.qkv_kernel_enabled
            ), "qkv_kernel_fuse_residual_add should be used with qkv_kernel_enabled"
            residual = attn_output.residual

        hidden_states = attn_output.hidden_states
        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if self.mlp_kernel_enabled and self.mlp_kernel_fused_rmsnorm:
                rmsnorm = self.post_attention_layernorm
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                rmsnorm = None
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=rmsnorm,
                adapter_ids=adapter_ids,
            )

        # if fuse residual add with qkv, we leave this add to the next layer's QKV
        # unless it is the last layer in which case we add it here
        if not self.qkv_kernel_fuse_residual_add:
            hidden_states = residual + hidden_states
            residual = None  # set to None to prevent it from being used again

        # also return residual for QKV in the next layer
        outputs = (hidden_states, attn_output.present_key_value, attn_output.cos_cache, attn_output.sin_cache, residual)
        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                sequence_dimension=1,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )

            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        updated_configs = get_updated_configs(config)

        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            # replicate fc weights since activations are sequence sharded
            self.fc = WeightGatheredColumnParallel(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True, sequence_dimension=1
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)


class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return LlamaForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""

        neuron_config = config.neuron_config
        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            """
            for every layer do the following transformations
            gate_w_prime = (gate_w.T * gamma).T
            up_w_prime = (up_w.T * gamma).T
            """
            if (
                neuron_config.fused_rmsnorm_skip_gamma
                and not neuron_config.sequence_parallel_enabled
            ):
                if neuron_config.mlp_kernel_enabled:
                    # MLP
                    state_dict[f"layers.{i}.mlp.gate_proj.weight"] = state_dict[
                        f"layers.{i}.mlp.gate_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.mlp.up_proj.weight"] = state_dict[
                        f"layers.{i}.mlp.up_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

                if neuron_config.qkv_kernel_enabled:
                    # QKV
                    state_dict[f"layers.{i}.self_attn.q_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.q_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.self_attn.k_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.k_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)
                    state_dict[f"layers.{i}.self_attn.v_proj.weight"] = state_dict[
                        f"layers.{i}.self_attn.v_proj.weight"
                    ] * state_dict[f"layers.{i}.input_layernorm.weight"].unsqueeze(0)

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig