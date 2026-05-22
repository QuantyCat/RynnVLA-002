import math
import os
import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import torch.nn as nn

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.pretrain import PretrainSolverBase_ck_action_head


def add_lora_to_model(model: nn.Module, lora_r: int, lora_alpha: int,
                      target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                      lora_dropout: float = 0.05,
                      dtype=torch.bfloat16,
                      train_lm_head: bool = True) -> None:
    """
    Add LoRA parameters directly to target nn.Linear modules.
    Avoids peft's module-wrapping approach which is incompatible with FSDP use_orig_params=True.
    Parameters are added via register_parameter so FSDP will correctly shard/gather them.
    """
    lora_scaling = lora_alpha / lora_r
    dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else None

    for name, module in model.named_modules():
        module_leaf = name.split(".")[-1]
        if not isinstance(module, nn.Linear) or module_leaf not in target_modules:
            continue

        in_features = module.weight.shape[1]
        out_features = module.weight.shape[0]
        param_device = module.weight.device  # keep LoRA on same device as the layer

        # LoRA A: initialized with Kaiming uniform, scaled down
        lora_A = nn.Parameter(torch.empty(lora_r, in_features, dtype=dtype, device=param_device))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        module.register_parameter("lora_weight_A", lora_A)

        # LoRA B: initialized to zeros so LoRA starts as identity
        lora_B = nn.Parameter(torch.zeros(out_features, lora_r, dtype=dtype, device=param_device))
        module.register_parameter("lora_weight_B", lora_B)

        module._lora_scaling = lora_scaling
        module._lora_dropout = dropout_layer

    # Register forward hooks after all parameters are added
    def make_lora_hook():
        def lora_hook(module, input, output):
            if not hasattr(module, "lora_weight_A"):
                return output
            x = input[0]
            if module._lora_dropout is not None and module.training:
                x_drop = module._lora_dropout(x)
            else:
                x_drop = x
            lora_out = (x_drop @ module.lora_weight_A.T) @ module.lora_weight_B.T
            return output + lora_out * module._lora_scaling
        return lora_hook

    for name, module in model.named_modules():
        if hasattr(module, "lora_weight_A"):
            module.register_forward_hook(make_lora_hook())

    # Freeze everything, then unfreeze LoRA/action head. lm_head is optional:
    # it is very large and usually unnecessary for continuous action tuning.
    for name, param in model.named_parameters():
        param.requires_grad = (
            "lora_weight_" in name
            or "action_head" in name
            or (train_lm_head and "lm_head" in name)
        )


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(PretrainSolverBase_ck_action_head):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_sign_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_sign_eps", type=float, default=0.03)
        parser.add_argument("--action_sign_margin", type=float, default=0.02)
        parser.add_argument("--action_wrong_sign_loss_multiplier", type=float, default=1.0)
        parser.add_argument("--action_wrong_sign_joint_weights", type=str, default=None)
        parser.add_argument("--action_sign_joint_weights", type=str, default=None)
        parser.add_argument("--action_sign_horizon_weights", type=str, default=None)
        parser.add_argument("--action_sign_center", type=str, default="raw_zero", choices=["raw_zero", "normalized_zero"])
        parser.add_argument("--action_quiet_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_quiet_eps", type=float, default=0.01)
        parser.add_argument("--action_quiet_pred_eps", type=float, default=0.01)
        parser.add_argument("--action_quiet_joint_weights", type=str, default=None)
        parser.add_argument("--action_quiet_horizon_weights", type=str, default=None)
        parser.add_argument("--action_motion_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_motion_eps", type=float, default=0.08)
        parser.add_argument("--action_motion_joint_weights", type=str, default=None)
        parser.add_argument("--action_motion_horizon_weights", type=str, default=None)
        parser.add_argument("--action_magnitude_loss_weight", type=float, default=0.0)
        parser.add_argument("--action_magnitude_eps", type=float, default=0.08)
        parser.add_argument("--action_magnitude_joint_weights", type=str, default=None)
        parser.add_argument("--action_magnitude_horizon_weights", type=str, default=None)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        parser.add_argument("--preprocess", default='true', choices=['true', 'false'])
        parser.add_argument("--with_state", action='store_true')
        parser.add_argument("--with_wrist", action='store_true')
        parser.add_argument("--with_action", action='store_true')
        parser.add_argument("--with_world_model", action='store_true')
        parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
        parser.add_argument("--tokenizer_path", type=str, default="../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af")
        parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 = disabled, use full fine-tuning)")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
        parser.add_argument("--train_lm_head", action="store_true", help="Keep lm_head trainable in LoRA mode")
        parser.add_argument(
            "--action_head_detach_hidden_states",
            action="store_true",
            help="Detach transformer hidden states before the continuous action head",
        )
        parser.add_argument(
            "--action_head_loss_routing",
            default="default",
            choices=["default", "sign_to_backbone"],
            help="Route selected action losses through attached hidden states",
        )
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):
        checkpoint_model_path = os.path.join(init_from, "model.safetensors")
        manual_lora_resume = (
            self.dp_rank == 0
            and getattr(self.args, "lora_r", 0) > 0
            and os.path.isfile(checkpoint_model_path)
        )
        model_init_from = self.args.init_from if manual_lora_resume else init_from

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        if self.dp_rank == 0:
            # For single-GPU training (dp_world_size == 1), load directly to CUDA to avoid
            # bf16 numerical instability observed when loading to CPU and then moving with .to().
            # For multi-GPU, keep CPU loading so FSDP can sync weights across ranks.
            _device_map = "cuda" if self.dp_world_size == 1 else "cpu"
            model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                model_init_from,
                action_dim=self.args.action_dim,
                time_horizon=self.args.time_horizon,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                action_sign_loss_weight=self.args.action_sign_loss_weight,
                action_sign_eps=self.args.action_sign_eps,
                action_sign_margin=self.args.action_sign_margin,
                action_wrong_sign_loss_multiplier=self.args.action_wrong_sign_loss_multiplier,
                action_wrong_sign_joint_weights=self.args.action_wrong_sign_joint_weights,
                action_sign_joint_weights=self.args.action_sign_joint_weights,
                action_sign_horizon_weights=self.args.action_sign_horizon_weights,
                action_sign_center=self.args.action_sign_center,
                action_quiet_loss_weight=self.args.action_quiet_loss_weight,
                action_quiet_eps=self.args.action_quiet_eps,
                action_quiet_pred_eps=self.args.action_quiet_pred_eps,
                action_quiet_joint_weights=self.args.action_quiet_joint_weights,
                action_quiet_horizon_weights=self.args.action_quiet_horizon_weights,
                action_motion_loss_weight=self.args.action_motion_loss_weight,
                action_motion_eps=self.args.action_motion_eps,
                action_motion_joint_weights=self.args.action_motion_joint_weights,
                action_motion_horizon_weights=self.args.action_motion_horizon_weights,
                action_magnitude_loss_weight=self.args.action_magnitude_loss_weight,
                action_magnitude_eps=self.args.action_magnitude_eps,
                action_magnitude_joint_weights=self.args.action_magnitude_joint_weights,
                action_magnitude_horizon_weights=self.args.action_magnitude_horizon_weights,
                action_head_detach_hidden_states=self.args.action_head_detach_hidden_states,
                action_head_loss_routing=self.args.action_head_loss_routing,
                torch_dtype=torch.bfloat16,
                device_map=_device_map,
            )
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    model_init_from,
                    action_dim=self.args.action_dim,
                    time_horizon=self.args.time_horizon,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    action_sign_loss_weight=self.args.action_sign_loss_weight,
                    action_sign_eps=self.args.action_sign_eps,
                    action_sign_margin=self.args.action_sign_margin,
                    action_wrong_sign_loss_multiplier=self.args.action_wrong_sign_loss_multiplier,
                    action_wrong_sign_joint_weights=self.args.action_wrong_sign_joint_weights,
                    action_sign_joint_weights=self.args.action_sign_joint_weights,
                    action_sign_horizon_weights=self.args.action_sign_horizon_weights,
                    action_sign_center=self.args.action_sign_center,
                    action_quiet_loss_weight=self.args.action_quiet_loss_weight,
                    action_quiet_eps=self.args.action_quiet_eps,
                    action_quiet_pred_eps=self.args.action_quiet_pred_eps,
                    action_quiet_joint_weights=self.args.action_quiet_joint_weights,
                    action_quiet_horizon_weights=self.args.action_quiet_horizon_weights,
                    action_motion_loss_weight=self.args.action_motion_loss_weight,
                    action_motion_eps=self.args.action_motion_eps,
                    action_motion_joint_weights=self.args.action_motion_joint_weights,
                    action_motion_horizon_weights=self.args.action_motion_horizon_weights,
                    action_magnitude_loss_weight=self.args.action_magnitude_loss_weight,
                    action_magnitude_eps=self.args.action_magnitude_eps,
                    action_magnitude_joint_weights=self.args.action_magnitude_joint_weights,
                    action_magnitude_horizon_weights=self.args.action_magnitude_horizon_weights,
                    action_head_detach_hidden_states=self.args.action_head_detach_hidden_states,
                    action_head_loss_routing=self.args.action_head_loss_routing,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration_ck_action_head(config)

        del model.model.vqmodel

        if self.dp_rank == 0 and getattr(self.args, "lora_r", 0) > 0:
            add_lora_to_model(
                model,
                lora_r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                lora_dropout=0.05,
                dtype=torch.bfloat16,
                train_lm_head=getattr(self.args, "train_lm_head", False),
            )

        if manual_lora_resume:
            from safetensors.torch import load_file

            checkpoint_state = {
                key.removeprefix("module."): value
                for key, value in load_file(checkpoint_model_path, device="cpu").items()
            }
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys while loading {checkpoint_model_path}: {unexpected_keys[:20]}")
            self.logger.info(
                "Loaded LoRA-aware checkpoint model state from %s with %d missing and %d unexpected keys",
                checkpoint_model_path,
                len(missing_keys),
                len(unexpected_keys),
            )

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run_with_eval_awm_w()
