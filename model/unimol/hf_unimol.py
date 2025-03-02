import argparse
import random
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedModel
from unicore.data import Dictionary

from .models import UniMolModel, unimol_base_architecture
from .utils.custom_utils import isnotebook


class UniMolConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim=512,
        inner_dim=512,
        num_classes=1,
        dropout=0.1,
        decoder_type="mlp",
        loss_type="mse", # mse | flooding | weighted_mse
        flooding_alpha=5.0,
        weight_alpha = 2.0,
        clamp_zero=True,
        dict_path="./dict.txt",
        backbone_path=None,
        return_representation=True,
        freeze_backbone=False,
        **kwargs,
    ):
        super(UniMolConfig, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.decoder_type = decoder_type
        self.loss_type = loss_type
        self.flooding_alpha = flooding_alpha
        self.weight_alpha = weight_alpha
        self.clamp_zero = clamp_zero
        self.dict_path = dict_path
        self.backbone_path = backbone_path
        self.return_representation = return_representation
        self.freeze_backbone = freeze_backbone


class MLPDecoder(nn.Module): 
    def __init__(self, input_dim, inner_dim, num_classes, dropout, layer_num=2):
        super(MLPDecoder, self).__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        
        self.inner_layers = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim) for _ in range(layer_num-2)]
        )
        
        self.activation_fn = nn.GELU()
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # xaiver init
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        for layer in self.inner_layers:
            x = self.dropout(x)
            x = self.activation_fn(layer(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NormalizedDecoder(nn.Module):
    # Ax + b
    def __init__(self, input_dim, inner_dim, num_classes, dropout):
        super(NormalizedDecoder, self).__init__()
        self.mean_param = nn.Parameter(torch.randn(num_classes))
        self.std_param = nn.Parameter(torch.randn(num_classes))

        # delta
        self.decoder = MLPDecoder(input_dim, inner_dim, num_classes, dropout)

    def forward(self, x):
        x = self.decoder(x)
        x = x * torch.exp(self.std_param) + self.mean_param
        return x


class UniMol(PreTrainedModel):
    config_class = UniMolConfig

    def __init__(
        self,
        config,
    ):
        super(UniMol, self).__init__(config=config)
        self.decoder_type = config.decoder_type
        if self.decoder_type == "normalize":
            self.decoder = NormalizedDecoder(
                config.input_dim, config.inner_dim, config.num_classes, config.dropout
            )
        elif self.decoder_type == "mlp":
            self.decoder = MLPDecoder(
                config.input_dim, config.inner_dim, config.num_classes, config.dropout, layer_num=3 if config.freeze_backbone else 2
            )
        self.config = config
        self.model = self.init_backbone()
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def load_backbone(self, model, dictionary):
        self.model = model
        self.dictionary = dictionary
    
    def init_backbone(self, *args, **kwargs):
        parser = argparse.ArgumentParser()
        dictionary = Dictionary().load(self.config.dict_path)
        dictionary.add_symbol("[MASK]", is_special=True)
        args_unimol, unknown = parser.parse_known_args()
        if unknown:
            print(f"Unimol backbone got unkonwn args: ", unknown)
        # change args_unimol given input args
        for key, value in kwargs.items():
            setattr(args_unimol, key, value)

        unimol_base_architecture(args_unimol)
        model_backbone = UniMolModel(args_unimol, dictionary=dictionary)
        
        if self.config.backbone_path is not None:
            weight = torch.load(self.config.backbone_path)
            if "model" in weight:
                model_weight = torch.load(self.config.backbone_path)["model"]
                model_backbone.load_state_dict(model_weight, strict=False)
            else:
                model_backbone.load_state_dict(weight, strict=False)
            print("Model backbone weight loaded")
        
        return model_backbone
    
    def forward(
        self,
        src_tokens,
        src_coord,
        src_edge_type,
        src_distance,
        target=None,
        smi_name=None,
    ) -> Dict[str, torch.Tensor]:
        # print first 10 parameters
    
        device = next(self.parameters()).device
        output = self.model(
            src_tokens=src_tokens.to(device),
            src_edge_type=src_edge_type.to(device),
            src_distance=src_distance.to(device),
            src_coord=src_coord.to(device),
        )
        (encoder_rep, encoder_pair_rep) = output
        x = encoder_rep[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.decoder(x)
        # normalize cls token
        cls_repr = encoder_rep[:, 0, :]
        cls_repr = cls_repr / torch.norm(cls_repr, dim=1, keepdim=True)
        

        loss = None
        preference_loss = None
        if target is not None:
            # clamp zero
            if self.config.clamp_zero:
                target = torch.clamp(target, min=0)
            if self.config.loss_type == "mse":
                loss = F.mse_loss(x.squeeze(-1), target.squeeze(-1))
            elif self.config.loss_type == "flooding":
                # loss flooding 
                mse = (x.squeeze(-1) - target.squeeze(-1))**2
                modified_mse = abs(mse - self.config.flooding_alpha)  + self.config.flooding_alpha
                loss = (modified_mse).mean()  # This is it
            elif self.config.loss_type == "weighted_mse":
                avg = target.squeeze(-1).mean()
                weight = torch.abs(target.squeeze(-1) / avg + 1)
                loss = weight * (x.squeeze(-1) - target.squeeze(-1)) ** 2
                loss = loss.mean()
            elif self.config.loss_type == "plus_one_weighted_mse":
                weight = torch.abs(target.squeeze(-1))  + 1
                loss = weight * (x.squeeze(-1) - target.squeeze(-1)) ** 2
                loss = loss.mean()
            elif "preference" in self.config.loss_type:
                use_log = "log" in self.config.loss_type
                mse_loss = F.mse_loss(x.squeeze(-1), target.squeeze(-1))
                # construct a list of pred sorted by value of target
                rank_rewards_list = []
                
                # rewards = self.reward_decoder(encoder_rep).squeeze(-1)
                # rewards = x.squeeze(-1)
                target = target.squeeze(-1)
                for i in range(target.size(0)):
                    rank_rewards = x[target.argsort()[i]]
                    # expand rank_rewards to a list of rank_rewards
                    rank_rewards_list.append(rank_rewards)
                if "normalize" in self.config.loss_type:
                        rank_rewards = torch.stack(rank_rewards_list)
                        rank_rewards = (rank_rewards - rank_rewards.mean()) / rank_rewards.std()
                        # convert back to list
                        rank_rewards_list = [rank_rewards[i] for i in range(rank_rewards.size(0))]
                if "sampling" in self.config.loss_type:
                    # K sampling from 6 to 36
                    K = random.randint(6, 36)
                    rank_rewards_list = random.sample(rank_rewards_list, K)
                preference_loss = self.preference_loss([rank_rewards_list], use_log)            
                if use_log:
                    preference_loss = preference_loss * 0.5
                loss = preference_loss + mse_loss
            else:
                raise NotImplementedError(f"Loss type {self.config.loss_type} not implemented")
        if self.config.return_representation:
            return {
                "loss": loss,
                "logits": x,
                "encoder_rep": cls_repr,
                "preference_loss": preference_loss if preference_loss is not None else torch.tensor(0.0).to(device),
            }
        else: 
            return {
                "loss": loss,
                "logits": x,
                "preference_loss": preference_loss if preference_loss is not None else torch.tensor(0.0).to(device),
            }
        
    def preference_loss(self, rank_rewards_list: List[List[torch.tensor]], take_log=False) -> torch.Tensor:
        device = self.parameters().__next__().device
        total_loss = torch.tensor(0.0, device=device)
        add_count = 0
        
        for rank_rewards in rank_rewards_list:
            rank_rewards = torch.stack(rank_rewards)
            diff_matrix = rank_rewards.unsqueeze(1) - rank_rewards.unsqueeze(0)
            if take_log:
                sigmoid_diff = torch.sigmoid(diff_matrix)
            else: 
                log_sigmoid = nn.LogSigmoid()
                sigmoid_diff = log_sigmoid(diff_matrix)
            
            
            upper_tri_indices = torch.triu_indices(rank_rewards.size(0), rank_rewards.size(0), 1)
            loss_terms = sigmoid_diff[upper_tri_indices[0], upper_tri_indices[1]]
            
            total_loss += loss_terms.sum()
            add_count += loss_terms.numel()
            
        avg_loss = total_loss / add_count
        avg_loss = avg_loss.mean()
        return -avg_loss

def init_unimol_backbone(weight_path, dict_path="./dict.txt", *args, **kwargs):
    parser = argparse.ArgumentParser()
    dictionary = Dictionary().load(dict_path)
    dictionary.add_symbol("[MASK]", is_special=True)
    args_unimol, unknown = parser.parse_known_args()
    if unknown:
        print(f"Unimol backbone got unkonwn args: ", unknown)
    # change args_unimol given input args
    for key, value in kwargs.items():
        setattr(args_unimol, key, value)

    unimol_base_architecture(args_unimol)
    model_backbone = UniMolModel(args_unimol, dictionary=dictionary)
    weight = torch.load(weight_path)
    if "model" in weight:
        model_weight = torch.load(weight_path)["model"]
        model_backbone.load_state_dict(model_weight, strict=False)
    else:
        model_backbone.load_state_dict(weight, strict=False)

    print("Model loaded")
    return model_backbone, dictionary
