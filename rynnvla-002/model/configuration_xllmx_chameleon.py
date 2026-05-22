import logging
from typing import List

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class ChameleonXLLMXConfig(ChameleonConfig):

    def __init__(
        self,
        z_loss_weight: float = 0.0,
        action_dim: int = 7,
        time_horizon: int = 5,
        action_sign_loss_weight: float = 0.0,
        action_sign_eps: float = 0.03,
        action_sign_margin: float = 0.02,
        action_wrong_sign_loss_multiplier: float = 1.0,
        action_wrong_sign_joint_weights=None,
        action_sign_joint_weights=None,
        action_sign_horizon_weights=None,
        action_sign_center: str = "raw_zero",
        action_quiet_loss_weight: float = 0.0,
        action_quiet_eps: float = 0.01,
        action_quiet_pred_eps: float = 0.01,
        action_quiet_joint_weights=None,
        action_quiet_horizon_weights=None,
        action_motion_loss_weight: float = 0.0,
        action_motion_eps: float = 0.08,
        action_motion_joint_weights=None,
        action_motion_horizon_weights=None,
        action_magnitude_loss_weight: float = 0.0,
        action_magnitude_eps: float = 0.08,
        action_magnitude_joint_weights=None,
        action_magnitude_horizon_weights=None,
        action_head_detach_hidden_states: bool = False,
        action_head_loss_routing: str = "default",
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        self.action_sign_loss_weight = action_sign_loss_weight
        self.action_sign_eps = action_sign_eps
        self.action_sign_margin = action_sign_margin
        self.action_wrong_sign_loss_multiplier = action_wrong_sign_loss_multiplier
        self.action_wrong_sign_joint_weights = action_wrong_sign_joint_weights
        self.action_sign_joint_weights = action_sign_joint_weights
        self.action_sign_horizon_weights = action_sign_horizon_weights
        self.action_sign_center = action_sign_center
        self.action_quiet_loss_weight = action_quiet_loss_weight
        self.action_quiet_eps = action_quiet_eps
        self.action_quiet_pred_eps = action_quiet_pred_eps
        self.action_quiet_joint_weights = action_quiet_joint_weights
        self.action_quiet_horizon_weights = action_quiet_horizon_weights
        self.action_motion_loss_weight = action_motion_loss_weight
        self.action_motion_eps = action_motion_eps
        self.action_motion_joint_weights = action_motion_joint_weights
        self.action_motion_horizon_weights = action_motion_horizon_weights
        self.action_magnitude_loss_weight = action_magnitude_loss_weight
        self.action_magnitude_eps = action_magnitude_eps
        self.action_magnitude_joint_weights = action_magnitude_joint_weights
        self.action_magnitude_horizon_weights = action_magnitude_horizon_weights
        self.action_head_detach_hidden_states = action_head_detach_hidden_states
        self.action_head_loss_routing = action_head_loss_routing
        super().__init__(
            **kwargs,
        )
