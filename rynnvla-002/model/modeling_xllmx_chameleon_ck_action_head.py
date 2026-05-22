import functools
import ast
import logging
import math
import os
from typing import List

import torch
from torch import nn

from .chameleon import ChameleonForConditionalGeneration
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


__all__ = ["ChameleonXLLMXForConditionalGeneration_ck"]


def _load_action_zero_norm(action_dim: int, center_mode: str = "raw_zero") -> list[float]:
    if center_mode == "normalized_zero":
        return [0.0] * action_dim
    if center_mode != "raw_zero":
        raise ValueError(
            f"action_sign_center must be 'raw_zero' or 'normalized_zero', got {center_mode!r}"
        )

    stats_file = os.environ.get("RYNNVLA_ACTION_STATS_FILE")
    if not stats_file:
        logger.warning(
            "RYNNVLA_ACTION_STATS_FILE is unset; action sign loss will fall back to normalized-zero centering"
        )
        return [0.0] * action_dim

    mins = []
    maxs = []
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("Dim "):
                    continue
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 3:
                    continue
                mins.append(float(parts[1]))
                maxs.append(float(parts[2]))
                if len(mins) == action_dim:
                    break
    except OSError as exc:
        logger.warning("Could not read action stats file %s: %s", stats_file, exc)
        return [0.0] * action_dim

    if len(mins) != action_dim:
        logger.warning(
            "Action stats file %s had %s dims, expected %s; disabling raw-zero sign centering",
            stats_file,
            len(mins),
            action_dim,
        )
        return [0.0] * action_dim

    values = []
    for min_value, max_value in zip(mins, maxs):
        denom = max(max_value - min_value, 1e-8)
        values.append(2.0 * (0.0 - min_value) / denom - 1.0)
    return values


def _action_sign_joint_weights(weights, action_dim: int) -> list[float]:
    if weights is None or weights == "":
        return [1.0] * action_dim

    if isinstance(weights, str):
        try:
            weights = ast.literal_eval(weights)
        except (SyntaxError, ValueError):
            weights = [float(part.strip()) for part in weights.split(",") if part.strip()]

    weights_list = [float(weight) for weight in weights]
    if len(weights_list) != action_dim:
        raise ValueError(
            f"action_sign_joint_weights must contain {action_dim} values, got {len(weights_list)}"
        )
    return weights_list


def _action_joint_weights(weights, action_dim: int, name: str) -> list[float]:
    if weights is None or weights == "":
        return [1.0] * action_dim

    if isinstance(weights, str):
        try:
            weights = ast.literal_eval(weights)
        except (SyntaxError, ValueError):
            weights = [float(part.strip()) for part in weights.split(",") if part.strip()]

    weights_list = [float(weight) for weight in weights]
    if len(weights_list) != action_dim:
        raise ValueError(
            f"{name} must contain {action_dim} values, got {len(weights_list)}"
        )
    return weights_list


def _action_horizon_weights(weights, time_horizon: int, name: str) -> list[float]:
    if weights is None or weights == "":
        return [1.0] * time_horizon

    if isinstance(weights, str):
        try:
            weights = ast.literal_eval(weights)
        except (SyntaxError, ValueError):
            weights = [float(part.strip()) for part in weights.split(",") if part.strip()]

    weights_list = [float(weight) for weight in weights]
    if len(weights_list) != time_horizon:
        raise ValueError(
            f"{name} must contain {time_horizon} values, got {len(weights_list)}"
        )
    return weights_list

class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        time_horizon=15,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*action_dim, hidden_dim=hidden_dim, output_dim=action_dim
        )
        
    def __call__(self, x):
        return self.predict_action(x)

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, self.time_horizon, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action

class ActionHead(nn.Module):
    def __init__(self, action_dim=7, time_horizon=8, hidden_size_factor=0.25, num_encoder_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.num_encoder_layers = num_encoder_layers
        self.hidden_size = 4096
        self.reduced_hidden_size = int(self.hidden_size * hidden_size_factor)
        self.action_token_embeddings = nn.Embedding(1, time_horizon * action_dim * self.hidden_size)
        nn.init.normal_(self.action_token_embeddings.weight, std=0.02)
        self.hidden_projection = nn.Linear(self.hidden_size, self.reduced_hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_hidden_size,
            nhead=4,
            dim_feedforward=self.reduced_hidden_size * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.reduced_hidden_size)
        )
        self.output_projection = L1RegressionActionHead(
            self.reduced_hidden_size, self.reduced_hidden_size, self.time_horizon, self.action_dim
        )
        
    def forward(self, hidden_states, input_ids, attention_mask=None, target_token_id=10004, eval=False):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - 从模型得到的hidden states
            input_ids: [batch_size, seq_len] - 对应的input_ids
            attention_mask: [batch_size, seq_len] - 注意力掩码（可选）
            target_token_id: int - 目标token id (默认10004)
        
        Returns:
            actions: 预测的动作序列
        """
        # # 检查输入是否有NaN
        # print("=== NaN Debug Info ===")
        # print(f"Input hidden_states has NaN: {torch.isnan(hidden_states).any()}")
        # if torch.isnan(hidden_states).any():
        #     print(f"NaN positions in hidden_states: {torch.isnan(hidden_states).nonzero()}")
        
        batch_size = hidden_states.shape[0]
        action_tokens = self.action_token_embeddings.weight.view(1, self.time_horizon * self.action_dim, self.hidden_size).expand(batch_size, -1, -1)
        
        # print(f"Action tokens has NaN: {torch.isnan(action_tokens).any()}")
        # if torch.isnan(action_tokens).any():
        #     print(f"NaN positions in action_tokens: {torch.isnan(action_tokens).nonzero()}")
        
        # 第一步：提取每一行第一个target_token_id之前的token的hidden states
        extracted_hidden_states = []
        extracted_attention_masks = []

        flag = True
        
        for i in range(batch_size):
            # 找到第一个target_token_id的位置
            target_positions = (input_ids[i] == target_token_id).nonzero(as_tuple=True)[0]
            if len(target_positions) > 1 or eval:
                # 取第一个target_token_id之前的所有token
                end_pos = target_positions[0].item()
            else:
                continue
            
            # print(f"Batch {i}: end_pos = {end_pos}")
            
            # 提取对应的hidden states
            extracted_hidden = hidden_states[i, :end_pos, :]  # [end_pos, hidden_size]
            # print(f"Batch {i}: extracted_hidden has NaN: {torch.isnan(extracted_hidden).any()}")
            extracted_hidden_states.append(extracted_hidden)
            
            # 提取对应的attention mask（如果提供）
            if attention_mask is not None:
                extracted_mask = attention_mask[i, :end_pos]
                extracted_attention_masks.append(extracted_mask)
        
        if len(extracted_hidden_states) == 0:
            extracted_hidden_states.append(hidden_states[0, 0:1, :])
            flag = False
                
        # 第二步：为每一行添加action_tokens
        combined_states_list = []
        combined_attention_masks = []
        max_length = 0
        
        for i in range(len(extracted_hidden_states)):
            # 将当前样本的hidden states与action tokens拼接
            combined_hidden = torch.cat([extracted_hidden_states[i], action_tokens[i]], dim=0)
            # print(f"Batch {i}: combined_hidden has NaN: {torch.isnan(combined_hidden).any()}")
            combined_states_list.append(combined_hidden)
            
            # 处理attention mask
            if attention_mask is not None:
                # 为action tokens创建mask (全为1)
                action_tokens_mask = torch.ones(self.time_horizon * self.action_dim, 
                                            device=attention_mask.device, dtype=attention_mask.dtype)
                combined_mask = torch.cat([extracted_attention_masks[i], action_tokens_mask], dim=0)
                combined_attention_masks.append(combined_mask)
            
            # 记录最大长度
            max_length = max(max_length, combined_hidden.shape[0])
                
        # 第三步：补全所有序列到相同长度
        padded_hidden_states = []
        padded_attention_masks = []
        
        for i in range(len(extracted_hidden_states)):
            current_length = combined_states_list[i].shape[0]
            if current_length < max_length:
                # 用零向量补全hidden states
                padding = torch.zeros(max_length - current_length, self.hidden_size, 
                                    device=hidden_states.device, dtype=hidden_states.dtype)
                padded_hidden = torch.cat([combined_states_list[i], padding], dim=0)
            else:
                padded_hidden = combined_states_list[i]
            
            # print(f"Batch {i}: padded_hidden has NaN: {torch.isnan(padded_hidden).any()}")
            padded_hidden_states.append(padded_hidden)
            
            # 补全attention mask
            if attention_mask is not None:
                current_mask_length = combined_attention_masks[i].shape[0]
                if current_mask_length < max_length:
                    mask_padding = torch.zeros(max_length - current_mask_length, 
                                            device=attention_mask.device, dtype=attention_mask.dtype)
                    padded_mask = torch.cat([combined_attention_masks[i], mask_padding], dim=0)
                else:
                    padded_mask = combined_attention_masks[i]
                padded_attention_masks.append(padded_mask)
        
        # 堆叠成batch
        processed_hidden_states = torch.stack(padded_hidden_states, dim=0)  # [batch_size, max_length, hidden_size]
        # print(f"Processed hidden_states has NaN: {torch.isnan(processed_hidden_states).any()}")
        
        if attention_mask is not None:
            processed_attention_mask = torch.stack(padded_attention_masks, dim=0)  # [batch_size, max_length]
        else:
            processed_attention_mask = torch.ones(len(extracted_hidden_states), processed_hidden_states.shape[1], 
                                                device=processed_hidden_states.device)
        
        # print(f"Processed attention_mask has NaN: {torch.isnan(processed_attention_mask).any()}")
        
        # 投影到较小的维度
        projected_states = self.hidden_projection(processed_hidden_states)
        # print(f"Projected states has NaN: {torch.isnan(projected_states).any()}")
        
        # 检查hidden_projection层的权重
        # if hasattr(self.hidden_projection, 'weight'):
        #     print(f"Hidden projection weight has NaN: {torch.isnan(self.hidden_projection.weight).any()}")
        #     if hasattr(self.hidden_projection, 'bias') and self.hidden_projection.bias is not None:
        #         print(f"Hidden projection bias has NaN: {torch.isnan(self.hidden_projection.bias).any()}")
        
        # 通过transformer encoder
        transformer_output = self.transformer_encoder(
            projected_states,
            src_key_padding_mask=(1 - processed_attention_mask).bool()
        )
        # print(f"Transformer output has NaN: {torch.isnan(transformer_output).any()}")
        
        # 检查transformer encoder的参数
        # for name, param in self.transformer_encoder.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"Transformer encoder parameter {name} has NaN")
        
        # 第四步：提取action tokens对应的输出
        action_outputs = []
        for i in range(len(extracted_hidden_states)):
            # 计算当前样本原始序列长度
            original_length = extracted_hidden_states[i].shape[0]
            # action tokens在transformer输出中的位置
            action_start = original_length
            action_end = action_start + self.time_horizon * self.action_dim
            
            # 边界检查
            if action_end > transformer_output.shape[1]:
                print(f"Warning: action_end ({action_end}) > sequence length ({transformer_output.shape[1]}) for batch {i}")
                action_end = transformer_output.shape[1]
            
            # 提取action tokens对应的输出
            action_output_i = transformer_output[i, action_start:action_end, :]  # [time_horizon * action_dim, reduced_hidden_size]
            # print(f"Batch {i}: action_output has NaN: {torch.isnan(action_output_i).any()}")
            action_outputs.append(action_output_i)
                
        # 将所有action outputs堆叠
        action_outputs_tensor = torch.stack(action_outputs, dim=0)  # [batch_size, time_horizon * action_dim, reduced_hidden_size]
        # print(f"Action outputs tensor has NaN: {torch.isnan(action_outputs_tensor).any()}")
        
        # 生成最终的动作预测
        actions = self.output_projection(action_outputs_tensor)
        actions = actions.reshape(-1, self.action_dim)
        # print(f"Final actions has NaN: {torch.isnan(actions).any()}")
        
        # 检查output_projection层的权重
        # if hasattr(self.output_projection, 'weight'):
        #     print(f"Output projection weight has NaN: {torch.isnan(self.output_projection.weight).any()}")
        #     if hasattr(self.output_projection, 'bias') and self.output_projection.bias is not None:
        #         print(f"Output projection bias has NaN: {torch.isnan(self.output_projection.bias).any()}")
        
        # print("=== End NaN Debug Info ===")
        
        return actions, flag




class ChameleonXLLMXForConditionalGeneration_ck_action_head(GenerationMixin, ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        self.init_input_ids = None
        # self.action_dim = 7
        # self.action_head = ActionHead(action_dim=self.action_dim, time_horizon=5, hidden_size_factor=0.25, num_encoder_layers=2)
        # self.action_dim = 6
        # self.action_head = ActionHead(action_dim=self.action_dim, time_horizon=20, hidden_size_factor=0.25, num_encoder_layers=2)
        self.action_dim = config.action_dim
        self.action_head = ActionHead(action_dim=config.action_dim, time_horizon=config.time_horizon, hidden_size_factor=0.25, num_encoder_layers=2)
        self.action_sign_loss_weight = getattr(config, "action_sign_loss_weight", 0.0)
        self.action_sign_eps = getattr(config, "action_sign_eps", 0.03)
        self.action_sign_margin = getattr(config, "action_sign_margin", 0.02)
        self.action_wrong_sign_loss_multiplier = getattr(config, "action_wrong_sign_loss_multiplier", 1.0)
        self.action_sign_center = getattr(config, "action_sign_center", "raw_zero")
        self.action_quiet_loss_weight = getattr(config, "action_quiet_loss_weight", 0.0)
        self.action_quiet_eps = getattr(config, "action_quiet_eps", 0.01)
        self.action_quiet_pred_eps = getattr(config, "action_quiet_pred_eps", 0.01)
        self.action_motion_loss_weight = getattr(config, "action_motion_loss_weight", 0.0)
        self.action_motion_eps = getattr(config, "action_motion_eps", 0.08)
        self.action_magnitude_loss_weight = getattr(config, "action_magnitude_loss_weight", 0.0)
        self.action_magnitude_eps = getattr(config, "action_magnitude_eps", 0.08)
        self.action_head_detach_hidden_states = getattr(config, "action_head_detach_hidden_states", False)
        self.action_head_loss_routing = getattr(config, "action_head_loss_routing", "default")
        self.time_horizon = config.time_horizon
        self.post_init()
        self.action_zero_norm_values = _load_action_zero_norm(self.action_dim, self.action_sign_center)
        self.action_sign_joint_weight_values = _action_sign_joint_weights(
            getattr(config, "action_sign_joint_weights", None),
            self.action_dim,
        )
        self.action_wrong_sign_joint_weight_values = _action_joint_weights(
            getattr(config, "action_wrong_sign_joint_weights", None),
            self.action_dim,
            "action_wrong_sign_joint_weights",
        )
        self.action_sign_horizon_weight_values = _action_horizon_weights(
            getattr(config, "action_sign_horizon_weights", None),
            self.time_horizon,
            "action_sign_horizon_weights",
        )
        self.action_quiet_joint_weight_values = _action_joint_weights(
            getattr(config, "action_quiet_joint_weights", None),
            self.action_dim,
            "action_quiet_joint_weights",
        )
        self.action_quiet_horizon_weight_values = _action_horizon_weights(
            getattr(config, "action_quiet_horizon_weights", None),
            self.time_horizon,
            "action_quiet_horizon_weights",
        )
        self.action_motion_joint_weight_values = _action_joint_weights(
            getattr(config, "action_motion_joint_weights", None),
            self.action_dim,
            "action_motion_joint_weights",
        )
        self.action_motion_horizon_weight_values = _action_horizon_weights(
            getattr(config, "action_motion_horizon_weights", None),
            self.time_horizon,
            "action_motion_horizon_weights",
        )
        self.action_magnitude_joint_weight_values = _action_joint_weights(
            getattr(config, "action_magnitude_joint_weights", None),
            self.action_dim,
            "action_magnitude_joint_weights",
        )
        self.action_magnitude_horizon_weight_values = _action_horizon_weights(
            getattr(config, "action_magnitude_horizon_weights", None),
            self.time_horizon,
            "action_magnitude_horizon_weights",
        )
        logger.info(
            "Action sign loss center=%s zero_norm=%s",
            self.action_sign_center,
            self.action_zero_norm_values,
        )
        self.action_loss_debug = {}
        

    def forward(self, input_ids=None, labels=None, training=False, att_mask=True, **kwargs):
        self.action_loss_debug = {}

        if not training:
            # import pdb; pdb.set_trace()
            if self.init_input_ids is None:
                self.init_input_ids = input_ids
            else:
                self.init_input_ids = torch.cat([self.init_input_ids, input_ids], dim=-1)
            if not att_mask:
                attention_mask = None
            else:
                attention_mask = self.generate_att_mask_3(self.init_input_ids)
                kwargs['attention_mask'] = attention_mask.squeeze()[-1:]

            output_attentions = kwargs.get("output_attentions", None)
            output_hidden_states = kwargs.get("output_hidden_states", None)
            return_dict = kwargs.get("return_dict", None)
            pixel_values = kwargs.get("pixel_values", None)
            position_ids = kwargs.get("position_ids", None)
            past_key_values = kwargs.get("past_key_values", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            use_cache = kwargs.get("use_cache", None)
            cache_position = kwargs.get("cache_position", None)

            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            # Generation only consumes the final token logits. Projecting the full
            # sequence to vocab and casting to fp32 is what was causing inference OOM.
            logits = self.lm_head(hidden_states[:, -1:, :]).float()

            if self.config.mask_image_logits:
                image_tokens = self.model.vocabulary_mapping.image_tokens
                logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # import pdb; pdb.set_trace()
        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        if not att_mask:
            attention_mask = None
        else:
            attention_mask = self.generate_att_mask_3(input_ids)

        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, attention_mask=attention_mask, **kwargs
        )

        c_loss = result[0]

        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[1]                   # [8, 1266, 65536]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits.float(), dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)
        
        if 'output_hidden_states' in kwargs:
            # c_loss, additional_loss_dict, logits, hidden_states, labels_c
            hidden_states = result[2][-1]  # [batch_size, seq_len, hidden_dim]
            sign_to_backbone = self.action_head_loss_routing == "sign_to_backbone"
            action_head_hidden_states = (
                hidden_states.detach()
                if self.action_head_detach_hidden_states or sign_to_backbone
                else hidden_states
            )
            
            # Main continuous path: optionally detached so regression-style
            # losses update only the action head.
            predicted_actions, actions_flag = self.action_head(
                hidden_states=action_head_hidden_states,
                input_ids=input_ids,
                attention_mask=None,
                target_token_id=10004
            )

            if actions_flag == False:
                self.action_loss_debug = {"action_seq_count": torch.tensor(0.0, device=self.device)}
                return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, predicted_actions.mean()*0

            sign_predicted_actions = predicted_actions
            if sign_to_backbone:
                # Sign path keeps hidden states attached so directional loss can
                # still shape LoRA/backbone representations.
                sign_predicted_actions, sign_actions_flag = self.action_head(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=None,
                    target_token_id=10004,
                )
                if sign_actions_flag == False:
                    self.action_loss_debug = {"action_seq_count": torch.tensor(0.0, device=self.device)}
                    return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, predicted_actions.mean()*0
            
            # print(f"Predicted actions shape: {predicted_actions.shape}")
            # print(f"Predicted actions: {predicted_actions}")

            labels_action_dis, sequences = self.get_action_hs_label(result[2][-1], labels)
            labels_action_ct = self.decode_token_ids_to_actions(labels_action_dis)

            base_loss_ct = torch.nn.functional.smooth_l1_loss(predicted_actions, labels_action_ct, beta=0.03)
            loss_ct = base_loss_ct
            sign_loss_value = predicted_actions.new_tensor(0.0)
            sign_loss_unweighted = predicted_actions.new_tensor(0.0)
            sign_active_frac = predicted_actions.new_tensor(0.0)
            sign_agreement = predicted_actions.new_tensor(0.0)
            sign_violation_frac = predicted_actions.new_tensor(0.0)
            sign_product_mean = predicted_actions.new_tensor(0.0)
            sign_product_min = predicted_actions.new_tensor(0.0)
            sign_weight_denominator = predicted_actions.new_tensor(0.0)
            wrong_sign_weight_denominator = predicted_actions.new_tensor(0.0)
            wrong_sign_frac = predicted_actions.new_tensor(0.0)
            per_joint_sign_agreement = predicted_actions.new_zeros(self.action_dim)
            per_joint_sign_count = predicted_actions.new_zeros(self.action_dim)
            quiet_loss_value = predicted_actions.new_tensor(0.0)
            quiet_active_frac = predicted_actions.new_tensor(0.0)
            quiet_violation_frac = predicted_actions.new_tensor(0.0)
            quiet_pred_abs = predicted_actions.new_tensor(0.0)
            quiet_weight_denominator = predicted_actions.new_tensor(0.0)
            per_joint_quiet_count = predicted_actions.new_zeros(self.action_dim)
            motion_loss_value = predicted_actions.new_tensor(0.0)
            motion_active_frac = predicted_actions.new_tensor(0.0)
            motion_weight_denominator = predicted_actions.new_tensor(0.0)
            per_joint_motion_count = predicted_actions.new_zeros(self.action_dim)
            magnitude_loss_value = predicted_actions.new_tensor(0.0)
            magnitude_under_ratio = predicted_actions.new_tensor(0.0)
            magnitude_pred_abs = predicted_actions.new_tensor(0.0)
            magnitude_label_abs = predicted_actions.new_tensor(0.0)
            magnitude_weight_denominator = predicted_actions.new_tensor(0.0)
            zero_norm = torch.tensor(
                self.action_zero_norm_values,
                device=predicted_actions.device,
                dtype=predicted_actions.dtype,
            )
            pred_centered = predicted_actions - zero_norm.view(1, -1)
            sign_pred_centered = sign_predicted_actions - zero_norm.view(1, -1)
            label_centered = labels_action_ct - zero_norm.view(1, -1)
            horizon_index = torch.arange(
                predicted_actions.shape[0],
                device=predicted_actions.device,
            ) % self.time_horizon
            if self.action_sign_loss_weight > 0:
                sign_mask = label_centered.abs() > self.action_sign_eps
                sign_active_frac = sign_mask.float().mean()
                if sign_mask.any():
                    target_sign = label_centered.sign().detach()
                    pred_sign = sign_pred_centered.sign()
                    sign_product = target_sign * sign_pred_centered
                    sign_matches = (pred_sign == target_sign) & sign_mask
                    sign_agreement = sign_matches.float().sum() / sign_mask.float().sum().clamp_min(1.0)
                    joint_counts = sign_mask.float().sum(dim=0)
                    per_joint_sign_count = joint_counts
                    per_joint_sign_agreement = sign_matches.float().sum(dim=0) / joint_counts.clamp_min(1.0)
                    sign_loss = torch.nn.functional.relu(
                        self.action_sign_margin - sign_product
                    )
                    wrong_sign_mask = (sign_product < 0) & sign_mask
                    active_sign_loss = sign_loss[sign_mask]
                    active_sign_product = sign_product[sign_mask]
                    sign_loss_unweighted = active_sign_loss.mean()
                    sign_violation_frac = (active_sign_loss > 0).float().mean()
                    wrong_sign_frac = wrong_sign_mask.float().sum() / sign_mask.float().sum().clamp_min(1.0)
                    sign_product_mean = active_sign_product.mean()
                    sign_product_min = active_sign_product.min()
                    sign_weights = torch.tensor(
                        self.action_sign_joint_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    ).view(1, -1).clamp_min(0)
                    sign_horizon_weights = torch.tensor(
                        self.action_sign_horizon_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    )[horizon_index].view(-1, 1).clamp_min(0)
                    weighted_sign_mask = sign_mask.to(dtype=predicted_actions.dtype) * sign_weights * sign_horizon_weights
                    wrong_sign_joint_weights = torch.tensor(
                        self.action_wrong_sign_joint_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    ).view(1, -1).clamp_min(0)
                    wrong_sign_extra = (
                        wrong_sign_mask.to(dtype=predicted_actions.dtype)
                        * wrong_sign_joint_weights
                        * max(float(self.action_wrong_sign_loss_multiplier) - 1.0, 0.0)
                    )
                    sign_loss_multiplier = 1.0 + wrong_sign_extra
                    weighted_sign_loss = sign_loss * weighted_sign_mask * sign_loss_multiplier
                    sign_weight_denominator = weighted_sign_mask.sum()
                    wrong_sign_weight_denominator = (weighted_sign_mask * wrong_sign_extra).sum()
                    if sign_weight_denominator > 0:
                        sign_loss_value = weighted_sign_loss.sum() / sign_weight_denominator.clamp_min(1e-6)
                    else:
                        sign_loss_value = sign_loss_unweighted
                    loss_ct = loss_ct + self.action_sign_loss_weight * sign_loss_value
            if self.action_quiet_loss_weight > 0:
                quiet_mask = label_centered.abs() <= self.action_quiet_eps
                quiet_active_frac = quiet_mask.float().mean()
                per_joint_quiet_count = quiet_mask.float().sum(dim=0)
                if quiet_mask.any():
                    quiet_joint_weights = torch.tensor(
                        self.action_quiet_joint_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    ).view(1, -1).clamp_min(0)
                    quiet_horizon_weights = torch.tensor(
                        self.action_quiet_horizon_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    )[horizon_index].view(-1, 1).clamp_min(0)
                    weighted_quiet_mask = quiet_mask.to(dtype=predicted_actions.dtype) * quiet_joint_weights * quiet_horizon_weights
                    quiet_weight_denominator = weighted_quiet_mask.sum()
                    if quiet_weight_denominator > 0:
                        pred_abs = pred_centered.abs()
                        quiet_excess = torch.nn.functional.relu(pred_abs - self.action_quiet_pred_eps)
                        quiet_loss_value = (quiet_excess * weighted_quiet_mask).sum() / quiet_weight_denominator.clamp_min(1e-6)
                        quiet_pred_abs = (pred_abs * weighted_quiet_mask).sum() / quiet_weight_denominator.clamp_min(1e-6)
                        quiet_violations = (pred_abs > self.action_quiet_pred_eps).to(dtype=predicted_actions.dtype)
                        quiet_violation_frac = (quiet_violations * weighted_quiet_mask).sum() / quiet_weight_denominator.clamp_min(1e-6)
                        loss_ct = loss_ct + self.action_quiet_loss_weight * quiet_loss_value
            if self.action_motion_loss_weight > 0:
                motion_mask = label_centered.abs() > self.action_motion_eps
                motion_active_frac = motion_mask.float().mean()
                per_joint_motion_count = motion_mask.float().sum(dim=0)
                if motion_mask.any():
                    motion_weights = torch.tensor(
                        self.action_motion_joint_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    ).view(1, -1).clamp_min(0)
                    motion_horizon_weights = torch.tensor(
                        self.action_motion_horizon_weight_values,
                        device=predicted_actions.device,
                        dtype=predicted_actions.dtype,
                    )[horizon_index].view(-1, 1).clamp_min(0)
                    weighted_motion_mask = motion_mask.to(dtype=predicted_actions.dtype) * motion_weights * motion_horizon_weights
                    motion_weight_denominator = weighted_motion_mask.sum()
                    if motion_weight_denominator > 0:
                        motion_loss = torch.nn.functional.smooth_l1_loss(
                            pred_centered,
                            label_centered,
                            beta=0.03,
                            reduction="none",
                        )
                        motion_loss_value = (motion_loss * weighted_motion_mask).sum() / motion_weight_denominator.clamp_min(1e-6)
                        loss_ct = loss_ct + self.action_motion_loss_weight * motion_loss_value
            if self.action_magnitude_loss_weight > 0:
                magnitude_mask = label_centered.abs() > self.action_magnitude_eps
                magnitude_joint_weights = torch.tensor(
                    self.action_magnitude_joint_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).view(1, -1).clamp_min(0)
                magnitude_horizon_weights = torch.tensor(
                    self.action_magnitude_horizon_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                )[horizon_index].view(-1, 1).clamp_min(0)
                magnitude_weights = magnitude_mask.to(dtype=predicted_actions.dtype) * magnitude_joint_weights * magnitude_horizon_weights
                magnitude_weight_denominator = magnitude_weights.sum()
                if magnitude_weight_denominator > 0:
                    magnitude_pred_abs = (pred_centered.abs() * magnitude_weights).sum() / magnitude_weight_denominator.clamp_min(1e-6)
                    magnitude_label_abs = (label_centered.abs() * magnitude_weights).sum() / magnitude_weight_denominator.clamp_min(1e-6)
                    magnitude_under = torch.nn.functional.relu(magnitude_label_abs - magnitude_pred_abs)
                    magnitude_under_ratio = magnitude_under / magnitude_label_abs.clamp_min(1e-6)
                    magnitude_loss_value = magnitude_under
                    loss_ct = loss_ct + self.action_magnitude_loss_weight * magnitude_loss_value

            self.action_loss_debug = {
                "action_seq_count": torch.tensor(float(len(sequences)), device=predicted_actions.device),
                "action_head_detach_hidden_states": torch.tensor(
                    float(self.action_head_detach_hidden_states),
                    device=predicted_actions.device,
                ),
                "action_loss_routing_sign_to_backbone": torch.tensor(
                    float(sign_to_backbone),
                    device=predicted_actions.device,
                ),
                "action_smooth_l1": base_loss_ct.detach(),
                "action_quiet_loss": quiet_loss_value.detach(),
                "action_quiet_weighted": (self.action_quiet_loss_weight * quiet_loss_value).detach(),
                "action_quiet_active_frac": quiet_active_frac.detach(),
                "action_quiet_violation_frac": quiet_violation_frac.detach(),
                "action_quiet_pred_abs": quiet_pred_abs.detach(),
                "action_quiet_eps": torch.tensor(float(self.action_quiet_eps), device=predicted_actions.device),
                "action_quiet_pred_eps": torch.tensor(float(self.action_quiet_pred_eps), device=predicted_actions.device),
                "action_quiet_weight_denominator": quiet_weight_denominator.detach(),
                "action_quiet_horizon_weight_sum": torch.tensor(
                    self.action_quiet_horizon_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_quiet_joint_weight_sum": torch.tensor(
                    self.action_quiet_joint_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_motion_loss": motion_loss_value.detach(),
                "action_motion_weighted": (self.action_motion_loss_weight * motion_loss_value).detach(),
                "action_motion_active_frac": motion_active_frac.detach(),
                "action_motion_eps": torch.tensor(float(self.action_motion_eps), device=predicted_actions.device),
                "action_motion_weight_denominator": motion_weight_denominator.detach(),
                "action_motion_horizon_weight_sum": torch.tensor(
                    self.action_motion_horizon_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_motion_joint_weight_sum": torch.tensor(
                    self.action_motion_joint_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_sign_loss": sign_loss_value.detach(),
                "action_sign_loss_unweighted": sign_loss_unweighted.detach(),
                "action_sign_weighted": (self.action_sign_loss_weight * sign_loss_value).detach(),
                "action_sign_active_frac": sign_active_frac.detach(),
                "action_sign_agreement": sign_agreement.detach(),
                "action_sign_violation_frac": sign_violation_frac.detach(),
                "action_sign_product_mean": sign_product_mean.detach(),
                "action_sign_product_min": sign_product_min.detach(),
                "action_sign_margin_value": torch.tensor(float(self.action_sign_margin), device=predicted_actions.device),
                "action_wrong_sign_loss_multiplier": torch.tensor(float(self.action_wrong_sign_loss_multiplier), device=predicted_actions.device),
                "action_wrong_sign_frac": wrong_sign_frac.detach(),
                "action_wrong_sign_extra_weight_denominator": wrong_sign_weight_denominator.detach(),
                "action_sign_weight_denominator": sign_weight_denominator.detach(),
                "action_sign_joint_weight_sum": torch.tensor(
                    self.action_sign_joint_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_sign_horizon_weight_sum": torch.tensor(
                    self.action_sign_horizon_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_wrong_sign_joint_weight_sum": torch.tensor(
                    self.action_wrong_sign_joint_weight_values,
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).sum().detach(),
                "action_magnitude_loss": magnitude_loss_value.detach(),
                "action_magnitude_weighted": (self.action_magnitude_loss_weight * magnitude_loss_value).detach(),
                "action_magnitude_under_ratio": magnitude_under_ratio.detach(),
                "action_magnitude_pred_abs": magnitude_pred_abs.detach(),
                "action_magnitude_label_abs": magnitude_label_abs.detach(),
                "action_magnitude_weight_denominator": magnitude_weight_denominator.detach(),
            }
            for joint_idx in range(self.action_dim):
                self.action_loss_debug[f"action_sign_agreement_j{joint_idx}"] = per_joint_sign_agreement[joint_idx].detach()
                self.action_loss_debug[f"action_sign_count_j{joint_idx}"] = per_joint_sign_count[joint_idx].detach()
                self.action_loss_debug[f"action_quiet_count_j{joint_idx}"] = per_joint_quiet_count[joint_idx].detach()
                self.action_loss_debug[f"action_motion_count_j{joint_idx}"] = per_joint_motion_count[joint_idx].detach()
                self.action_loss_debug[f"action_sign_center_j{joint_idx}"] = torch.tensor(
                    self.action_zero_norm_values[joint_idx],
                    device=predicted_actions.device,
                    dtype=predicted_actions.dtype,
                ).detach()
            focus_joint_idx = min(3, self.action_dim - 1)
            for horizon_idx in range(self.time_horizon):
                horizon_mask = horizon_index == horizon_idx
                if not horizon_mask.any():
                    continue
                pred_hj = pred_centered[horizon_mask, focus_joint_idx]
                label_hj = label_centered[horizon_mask, focus_joint_idx]
                pred_hj_norm0 = predicted_actions[horizon_mask, focus_joint_idx]
                label_hj_norm0 = labels_action_ct[horizon_mask, focus_joint_idx]
                active_hj = label_hj.abs() > self.action_motion_eps
                active_norm0_hj = label_hj_norm0.abs() > self.action_sign_eps
                prefix = f"action_h{horizon_idx + 1}_j{focus_joint_idx}"
                self.action_loss_debug[f"{prefix}_centered_pred_abs"] = pred_hj.abs().mean().detach()
                self.action_loss_debug[f"{prefix}_centered_label_abs"] = label_hj.abs().mean().detach()
                self.action_loss_debug[f"{prefix}_centered_l1"] = torch.nn.functional.smooth_l1_loss(
                    pred_hj,
                    label_hj,
                    beta=0.03,
                ).detach()
                if active_hj.any():
                    self.action_loss_debug[f"{prefix}_centered_mag_ratio"] = (
                        pred_hj[active_hj].abs().mean() / label_hj[active_hj].abs().mean().clamp_min(1e-6)
                    ).detach()
                    self.action_loss_debug[f"{prefix}_centered_sign"] = (
                        pred_hj[active_hj].sign() == label_hj[active_hj].sign()
                    ).float().mean().detach()
                else:
                    self.action_loss_debug[f"{prefix}_centered_mag_ratio"] = predicted_actions.new_tensor(0.0)
                    self.action_loss_debug[f"{prefix}_centered_sign"] = predicted_actions.new_tensor(0.0)
                if active_norm0_hj.any():
                    self.action_loss_debug[f"{prefix}_norm0_sign"] = (
                        pred_hj_norm0[active_norm0_hj].sign() == label_hj_norm0[active_norm0_hj].sign()
                    ).float().mean().detach()
                else:
                    self.action_loss_debug[f"{prefix}_norm0_sign"] = predicted_actions.new_tensor(0.0)

            # print(f"Predicted actions shape: {predicted_actions.shape}", f"GT actions shape: {labels_action_ct.shape}")

            # import pdb; pdb.set_trace()
            
            return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, loss_ct
        else:
            return c_loss, additional_loss_dict
    
    def get_action_hs_label(self, hidden_states, labels_c):

        # 找到所有符合条件的序列
        sequences = self.find_sequences(labels_c)

        # 初始化结果张量
        labels_action = torch.zeros(len(sequences), self.action_dim, dtype=torch.long, device=self.device)
        
        # 填充结果张量
        for i, (batch, start) in enumerate(sequences):
            labels_action[i] = labels_c[batch, start:start+self.action_dim]
        
        return labels_action, sequences
    
    def find_sequences(self, tensor_input):
        # 找到所有以 10004 开始，15005 结束的序列
        start_indices = (tensor_input[:, :-1*self.action_dim+1] == 10004).nonzero(as_tuple=True)
        valid_sequences = []
        for batch, start in zip(*start_indices):
            if tensor_input[batch, start+self.action_dim+1] == 15004:
                valid_sequences.append((batch, start+1))
        return valid_sequences


    def generate_att_mask_3(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # 创建初始的下三角矩阵作为基础注意力掩码
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).bool()
        
        # 找到所有特殊标记的位置
        image_start = (input_ids == 8197)  # 图像块开始标记
        image_end = (input_ids == 8196)    # 图像块结束标记
        action_start = (input_ids == 10004)  # 动作块开始标记
        action_end = (input_ids == 15004)    # 动作块结束标记

        # 找到每个batch中所有的图像块和动作块的起始和结束位置
        image_blocks = []
        action_blocks = []
        for batch_idx in range(batch_size):
            # 找到当前batch的图像块起始和结束位置
            image_starts = torch.where(image_start[batch_idx])[0]
            image_ends = torch.where(image_end[batch_idx])[0]
            
            # 如果图像块的起始和结束位置不匹配
            if len(image_starts) > len(image_ends):
                # 将当前batch的最后一个位置作为缺失的结束标记
                last_position = seq_len - 1
                image_ends = torch.cat([image_ends, torch.tensor([last_position], dtype=torch.long, device=self.device)])
            elif len(image_starts) < len(image_ends):
                image_ends = image_ends[:-1]
            
            # 确保图像块的起始和结束位置匹配
            if len(image_starts) != len(image_ends):
                raise ValueError("Mismatched image start and end tokens in batch.")
            
            # 存储图像块的起始和结束位置
            image_blocks.append(list(zip(image_starts.cpu().numpy(), image_ends.cpu().numpy())))
            
            # 找到当前batch的动作块起始和结束位置
            action_starts = torch.where(action_start[batch_idx])[0]
            action_ends = torch.where(action_end[batch_idx])[0]
            
            # 如果动作块的起始和结束位置不匹配
            if len(action_starts) > len(action_ends):
                # 将当前batch的最后一个位置作为缺失的结束标记
                last_position = seq_len - 1
                action_ends = torch.cat([action_ends, torch.tensor([last_position], dtype=torch.long, device=self.device)])
            elif len(action_starts) < len(action_ends):
                action_ends = action_ends[:-1]
            
            # 确保动作块的起始和结束位置匹配
            if len(action_starts) != len(action_ends):
                raise ValueError("Mismatched action start and end tokens in batch.")
            
            # 存储动作块的起始和结束位置
            action_blocks.append(list(zip(action_starts.cpu().numpy(), action_ends.cpu().numpy())))

        # 遍历每个batch并更新mask
        for batch_idx in range(batch_size):
            # 获取当前batch的图像块和动作块
            current_image_blocks = image_blocks[batch_idx]
            current_action_blocks = action_blocks[batch_idx]
            
            # 找到最后一个图像块的结束位置
            if current_image_blocks:
                last_image_end = current_image_blocks[-1][1]  # 最后一个图像块的结束位置
            else:
                last_image_end = -1  # 如果没有图像块，则认为所有动作块都在图像块之后
            
            # 遍历当前batch的所有动作块
            for block_start, block_end in current_action_blocks:
                # 判断当前动作块是否在最后一个图像块之后
                if block_start > last_image_end:
                    # 找到当前动作块之前的所有动作块
                    previous_action_blocks = [
                        (s, e) for s, e in current_action_blocks if e < block_start
                    ]
                    
                    # 如果存在之前的动作块，将当前动作块与这些动作块之间的注意力设为0
                    for prev_start, prev_end in previous_action_blocks:
                        mask[batch_idx, block_start:block_end + 1, prev_start:prev_end + 1] = 0
                else:
                    # 如果当前动作块不在最后一个图像块之后，则保持注意力为1
                    pass  # 默认情况下已经是1，无需额外操作
        
        return mask
    
    def generate_img(self, input_ids, generation_config):
        # res = ChameleonForConditionalGeneration.generate(
        #     self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True, use_cache=True, past_key_values=past_key_values
        # )

        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True, att_mask=None
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:]
        # dis_tokens = res['sequences']
        # import pdb; pdb.set_trace()
        return dis_tokens
    
    def generate_dis_ma(self, input_ids, generation_config):
        self.init_input_ids = None
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config, output_hidden_states=True, training=False, return_dict_in_generate=True
        )
        dis_tokens = res['sequences'][:, input_ids.shape[1]:][0]
        # print(dis_tokens)
        decoded_actions = self.decode_token_ids_to_actions(dis_tokens)

        action_sequences = []
        for i, token in enumerate(dis_tokens):
            if token == 10004:
                start_index = i
            elif token == 15004:
                end_index = i
                if start_index is not None:
                    action_sequences.append(decoded_actions[start_index+1:end_index])
                start_index = None
                
        return action_sequences
    
    def generate_action_head(self, input_ids, generation_config):
        """
        生成一个token（期望为10004），然后使用action_head预测动作
        """
        self.init_input_ids = None
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if (input_ids == 10004).any():
            full_attention_mask = self.generate_att_mask_3(input_ids)
            result = ChameleonForConditionalGeneration.forward(
                self,
                input_ids=input_ids,
                labels=None,
                use_cache=False,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = result.hidden_states[-1]
            predicted_actions, actions_flag = self.action_head(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=None,
                target_token_id=10004,
                eval=True,
            )
            if not actions_flag:
                print("Warning: Action prediction failed, returning zero actions")
                return torch.zeros(self.action_head.time_horizon, self.action_head.action_dim, device=input_ids.device)

            predicted_actions = predicted_actions.reshape(self.action_head.time_horizon, self.action_head.action_dim)
            print(f"Predicted actions shape: {predicted_actions.shape}")
            print(f"Predicted actions: {predicted_actions}")
            return predicted_actions
        
        # 生成一个token（期望为10004）
        res = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=8710,
            output_hidden_states=True,
            training=False,
            return_dict_in_generate=True,
        )
        
        # 获取生成的token
        generated_token = res['sequences'][:, input_ids.shape[1]:]  # [batch_size, 1]
        print(f"Generated token: {generated_token}")
        
        # 构建完整的input_ids（原始输入 + 生成的token）
        full_input_ids = res['sequences']  # [batch_size, original_length + 1]
        
        # 获取最后一步生成token对应的hidden states
        new_token_hidden_states_list = [
            step_hidden_states[-1] for step_hidden_states in res['hidden_states']
        ]
        last_step_hidden_states = torch.cat(new_token_hidden_states_list, dim=1)
        
        # 使用action_head预测动作
        predicted_actions, actions_flag = self.action_head(
            hidden_states=last_step_hidden_states,
            input_ids=full_input_ids,
            attention_mask=None,
            target_token_id=10004,
            eval=True
        )
        
        # 检查是否成功预测动作
        if not actions_flag:
            print("Warning: Action prediction failed, returning zero actions")
            return torch.zeros(self.action_head.time_horizon, self.action_head.action_dim, device=input_ids.device)
        
        # 将predicted_actions重新reshape为[time_horizon, action_dim]
        predicted_actions = predicted_actions.reshape(self.action_head.time_horizon, self.action_head.action_dim)
        
        print(f"Predicted actions shape: {predicted_actions.shape}")
        print(f"Predicted actions: {predicted_actions}")
        
        return predicted_actions



    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens, self.action_head]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        return modules
    
    def decode_token_ids_to_actions(self, dis_action):
        bins = torch.linspace(-1, 1, 256, device=dis_action.device)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        discretized_actions = dis_action - 1 - 10004
        discretized_actions = torch.clamp(discretized_actions - 1, min=0, max=bin_centers.shape[0] - 1).long()
        return bin_centers[discretized_actions]
