"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
import numpy as np
from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=None,  # YOPOv2-Tracker paper: 6D state = v_xyz, a_xyz
            output_dim=None,  # 10: x_pva, y_pva, z_pva, score; 14 adds objectness and target u/v/depth
            hidden_state=64,
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        observation_dim = int(cfg.get("observation_dim", 6) if observation_dim is None else observation_dim)
        output_dim = int(cfg.get("output_dim", 10) if output_dim is None else output_dim)
        input_channels = int(cfg.get("image_channels", 1))
        self.output_dim = output_dim

        self.image_backbone = YopoBackbone(hidden_state, input_channels=input_channels)
        self.state_backbone = nn.Sequential()
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = torch.cat((obs_feature, depth_feature), 1)
        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])  # [batch, 9, vertical_num, horizon_num]
        score = torch.nn.functional.softplus(output[:, 9])  # [batch, vertical_num, horizon_num]
        if self.output_dim <= 10:
            return endstate, score
        objectness_logit = output[:, 10]
        target_raw = output[:, 11:14]
        return endstate, score, objectness_logit, target_raw

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: end state in the body frame
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        output = self.forward(depth, obs)
        if len(output) == 2:
            endstate_pred, score_pred = output
            objectness, target_uvd = None, None
        else:
            endstate_pred, score_pred, objectness_logit, target_raw = output
            objectness = torch.sigmoid(objectness_logit)
            target_uvd = self.state_transform.decode_target(target_raw)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        if objectness is None:
            return endstate, score_pred
        return endstate, score_pred, objectness, target_uvd

    def print_grad(self, grad):
        print("grad of hook: ", grad)
