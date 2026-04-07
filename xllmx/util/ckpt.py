import json
import logging
import os
import shutil
from typing import Dict, Optional
import subprocess

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

logger = logging.getLogger(__name__)


def split_ckpt_str_into_epoch_iter(ckpt_str: str):
    # divide ckpt directory names into epoch and iter parts
    parts = ckpt_str.split("-")
    epoch = int(parts[0].replace("epoch", ""))
    if len(parts) == 2:
        iter_part = int(parts[1].replace("iter", ""))
    else:
        iter_part = None
    return epoch, iter_part


def remove_early_ckpts(out_dir, max_keep=2):

    if max_keep <= 0:
        return

    def ckpt_sort_key(s):
        # divide ckpt directory names into epoch and iter parts
        epoch, iteration = split_ckpt_str_into_epoch_iter(s)
        if iteration is None:
            iteration = float("inf")
        return epoch, iteration

    existing_checkpoints = [_ for _ in os.listdir(out_dir) if "epoch" in _]
    existing_checkpoints = sorted(existing_checkpoints, key=ckpt_sort_key, reverse=True)

    for dir_to_remove in existing_checkpoints[max_keep:]:
        dir_to_remove = os.path.join(out_dir, dir_to_remove)
        shutil.rmtree(dir_to_remove)
        logger.info(f"Deleted {dir_to_remove}")


def save(
    output_dir,
    is_main_process,
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer=None,
    args=None,
    epoch=None,
    iteration=None,
    additional_rank_common: Optional[Dict] = None,
    additional_rank_specific: Optional[Dict] = None,
    max_keep=2,
):
    save_name = f"epoch{epoch}"
    if iteration is not None:
        save_name += f"-iter{iteration}"
    save_dir = os.path.join(output_dir, save_name)

    os.makedirs(save_dir, exist_ok=True)

    # save model
    _is_fsdp = isinstance(model, FSDP)

    def _save_model():
        save_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float,
        }[args.precision]

        consolidated_model_state_dict = {
            key: val.to(save_dtype) for key, val in model.state_dict().items()
        }

        if is_main_process:
            # Use safe_save_pretrained to avoid HF tied-weights bugs on single GPU
            inner = getattr(model, "module", model)
            if hasattr(inner, "save_pretrained"):
                # Temporarily clear _tied_weights_keys if it causes issues
                _orig = getattr(type(inner), "_tied_weights_keys", None)
                _inst_orig = inner.__dict__.get("_tied_weights_keys", _SENTINEL := object())
                try:
                    inner._tied_weights_keys = []
                    inner.save_pretrained(save_dir, state_dict=consolidated_model_state_dict)
                except Exception:
                    # Restore and retry via torch.save
                    torch.save(consolidated_model_state_dict, os.path.join(save_dir, "model.pt"))
                    logger.warning("save_pretrained failed, saved state dict to model.pt")
                finally:
                    if _inst_orig is not _SENTINEL:
                        inner._tied_weights_keys = _inst_orig
                    elif "_tied_weights_keys" in inner.__dict__:
                        del inner._tied_weights_keys
            else:
                torch.save(consolidated_model_state_dict, os.path.join(save_dir, "model.pt"))

    if _is_fsdp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            _save_model()
    else:
        _save_model()
    logger.info("model saved")

    # save optimizer
    if optimizer is not None:
        opt_path = os.path.join(
            save_dir,
            f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
        )
        if _is_fsdp:
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                torch.save(optimizer.state_dict(), opt_path)
        else:
            torch.save(optimizer.state_dict(), opt_path)
        logger.info("optimizer saved")
    else:
        logger.info("optimizer is None, skip saving")

    if additional_rank_specific is not None:
        torch.save(
            additional_rank_specific,
            os.path.join(save_dir, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"),
        )
        logger.info(f"additional_rank_specific {list(additional_rank_specific.keys())} saved")

    if not is_main_process:
        dist.barrier()
        return

    # =========The followings are for main process only=========
    if tokenizer is not None:
        tokenizer.save(save_dir)
        logger.info("tokenizer saved")
    else:
        logger.info("tokenizer is None, skip saving")

    if args is not None:
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("args saved")
    else:
        logger.info("args is None, skip saving")

    if additional_rank_common is not None:
        torch.save(additional_rank_common, os.path.join(save_dir, "additional_rank_common.pth"))
        logger.info(f"additional_resources {list(additional_rank_common.keys())} saved")

    remove_early_ckpts(output_dir, max_keep=max_keep)

    dist.barrier()
    return


