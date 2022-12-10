import numpy as np
import torch
from typing import Dict, List


DEFAULT_LOSS_WEIGHTS = np.array([1, 1, 1, 1, 10, 10, 10, 10])


def weighted_feature_matching_loss(y_true, fake_samples, image_input, real_samples, D, inner_weight, 
                          sample_weight):
    y_fake = D([image_input, fake_samples])[1:]
    y_real = D([image_input, real_samples])[1:]

    fm_loss = 0
    for i in range(len(y_fake)):
        if i<3:
            fm_loss += inner_weight * K.mean(K.abs(y_fake[i] - y_real[i]))
        else:
            fm_loss += (1-inner_weight) * K.mean(K.abs(y_fake[i] - y_real[i]))
    fm_loss *= sample_weight
    return fm_loss


class RVGANLoss:
    def __init__(self,
                 loss_weights: np.array = DEFAULT_LOSS_WEIGHTS,
                 inner_weight: float = 0.5):
        self.loss_weights = loss_weights
        self.up_inner_weight = inner_weight
        self.down_inner_weight = (1 - inner_weight)
        self.hinge_loss_1 = torch.nn.HingeEmbeddingLoss()
        self.hinge_loss_2 = torch.nn.HingeEmbeddingLoss()
        self.hinge_loss_3 = torch.nn.HingeEmbeddingLoss()
        self.hinge_loss_4 = torch.nn.HingeEmbeddingLoss()
        self.mse_1 = torch.nn.MSELoss()
        self.mse_2 = torch.nn.MSELoss()
        
    def compute_feature_loss(self,
                             real_discriminator_features: List,
                             fake_discriminator_features) -> float:
        num_down = int(len(real_discriminator_features) / 2)
        total_feature_loss = 0
        for i, (real_feat, fake_feat) in enumerate(zip(real_discriminator_features,
                                                       fake_discriminator_features)):
            feature_loss = (real_feat - fake_feat).abs().mean()
            if i < num_down:
                feature_loss *= self.up_inner_weight
            else:
                feature_loss *= self.down_inner_weight
            total_feature_loss += feature_loss

        return total_feature_loss

    def compute_rvgan_loss(self, model_output_dict: Dict) -> float:
        total_rvgan_loss = self.hinge_loss_1(
            model_output_dict["Fake"]["Coarse Discriminator Out"],
            -torch.ones_like(model_output_dict["Fake"]["Coarse Discriminator Out"])
        ).sum() * self.loss_weights[0]
        total_rvgan_loss += self.hinge_loss_2(
            model_output_dict["Fake"]["Fine Discriminator Out"],
            -torch.ones_like(model_output_dict["Fake"]["Fine Discriminator Out"])
        ).sum() * self.loss_weights[1]
        total_rvgan_loss += self.compute_feature_loss(
            model_output_dict["Real"]["Coarse Discriminator Features"],
            model_output_dict["Fake"]["Coarse Discriminator Features"]
        ) * self.loss_weights[2]
        total_rvgan_loss += self.compute_feature_loss(
            model_output_dict["Real"]["Fine Discriminator Features"],
            model_output_dict["Fake"]["Fine Discriminator Features"]
        ) * self.loss_weights[3]
        total_rvgan_loss += self.hinge_loss_3(
            model_output_dict["Fake"]["Coarse Discriminator Out"],
            model_output_dict["Vessel Labels"]["Coarse"]
        ).sum() * self.loss_weights[4]
        total_rvgan_loss += self.hinge_loss_4(
            model_output_dict["Fake"]["Fine Discriminator Out"],
            model_output_dict["Vessel Labels"]["Fine"]
        ).sum() * self.loss_weights[5]
        total_rvgan_loss += self.mse_1(
            model_output_dict["Fake"]["Coarse Discriminator Out"],
            model_output_dict["Vessel Labels"]["Coarse"]
        ).sum() * self.loss_weights[6]
        total_rvgan_loss += self.mse_2(
            model_output_dict["Fake"]["Fine Discriminator Out"],
            model_output_dict["Vessel Labels"]["Fine"]
        ).sum() * self.loss_weights[7]

        return total_rvgan_loss
