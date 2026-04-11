import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import numpy as np

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration_ck_action_head
from xllmx.solvers.pretrain import PretrainSolverBase

import tqdm
from PIL import Image


from lerobot_util.Chameleon_utils import get_action_Chameleon_dis_awm_ck, get_action_Chameleon_dis_awm_ck_wrist_action_head
from data_lerobot.pre_tokenize_action_state import ItemProcessor
import time
import xllmx.util as util
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter



class Solver(PretrainSolverBase):
    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard"))
        _rynnvla_dir = os.path.dirname(os.path.abspath(__file__))
        _tokenizer_path = os.path.join(_rynnvla_dir, "ckpts", "chameleon", "base_model")
        self.item_processor = ItemProcessor(tokenizer=_tokenizer_path, target_size=256)
        print('init done 000000!')
        self.his_img = []
        self.model, _ = self._model_func(self.args.resume_path)
        self.model.eval()
        print('init done!')


    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        parser.add_argument("--task_suite_name", type=str, default="libero_spatial",)
        parser.add_argument("--device", default=0, type=int, help="gpu device")
        parser.add_argument("--head", type=str, default="dis", choices=["dis", "ct"])
        parser.add_argument("--his", type=str, default="1h_1a", choices=["1h_1a", "2h_1a", "4h_1a", "2h_2a", "4h_4a", "1h_1a_img_only", "2h_1a_img_only", "4h_1a_img_only", "1h_1a_img_only_state",])
        parser.add_argument("--action_steps", default=25, type=int, help="actions to be excuted when multiple actions are generated")
        parser.add_argument("--half", default=0, type=int, help="which part of test set will be evaluated")
        parser.add_argument("--port", default=8000, type=int)
        parser.add_argument("--token", default='', type=str)
        parser.add_argument("--env", default='lerobot', type=str)
        parser.add_argument("--record", default=False, type=bool)
        parser.add_argument("--pack", default="protobuf", type=str)
        parser.add_argument("--action_rate", default=30, type=int)
        parser.add_argument("--compress", default='gzip', type=str)
        parser.add_argument("--action_dim", type=int, default=7)
        parser.add_argument("--time_horizon", type=int, default=5)
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration_ck_action_head, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840

        model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            init_from,
            action_dim=self.args.action_dim,
            time_horizon=self.args.time_horizon,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{self.args.device}",
            ignore_mismatched_sizes=True,
        )

        return model, None

    def _item_processor_func(self) -> ItemProcessor:
        return ItemProcessor(target_size=288)

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
        
    def unnorm_min_max(self, action):

        min_values = np.array([
            -0.13845688,  # dim 0
            -0.17819679,  # dim 1
            -0.19286394,  # dim 2
            -0.17750373,  # dim 3
            -0.28787115,  # dim 4
             0.00000000   # dim 5
        ])

        max_values = np.array([
            0.16925158,   # dim 0
            0.17430055,   # dim 1
            0.16337049,   # dim 2
            0.20580864,   # dim 3
            0.29125005,   # dim 4
            0.48936170    # dim 5
        ])     
            
        unnorm_action = (action + 1) / 2 * (max_values - min_values + 1e-8) + min_values
        
        return unnorm_action

    def get_action_wrist_action_head_state(self, front_image, wrist_image, state, prompt):

        # front_image from the front camera, type: numpy.ndarray, uint8, shape: (H, W, 3)
        # wrist_image from the wrist camera, type: numpy.ndarray, uint8, shape: (H, W, 3)
        # prompt: "Place the strawberries from the table into the cup."
        # state: state of the robot, shape: (6, ) for lerobot

        dis_action = get_action_Chameleon_dis_awm_ck_wrist_action_head(
                self.model,
                front_image,
                wrist_image,
                prompt,
                self.item_processor,
                self.his_img,
                self.args.his,
                self.args.action_steps,
                state
            )
        dis_action = dis_action.cpu().float().detach().numpy()
        
        dis_action_unnorm = self.unnorm_min_max(dis_action)

        self.his_img = [front_image]

        return dis_action_unnorm
