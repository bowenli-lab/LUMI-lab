# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.distributed.utils import all_gather
import torch.distributed as dist
from torch import nn
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out




class NT_Xent(nn.Module):
    def __init__(self, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.is_init = False
        self.temperature = temperature
        self.world_size = self.get_world_size()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _init(self, sample):
        self.world_size = self.get_world_size()
        self.device = sample.device
        self.batch_size = sample.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size, self.world_size)
        self.is_init = True
        
        
    def get_world_size(self, group=None):
        if torch.distributed.is_initialized():
            return dist.get_world_size(group=group)
        else:
            return 1

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if not self.is_init or z_i.size(0) != self.batch_size:
            self._init(z_i)
        
        N = 2 * self.batch_size * self.get_world_size()

        if self.get_world_size() > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
        
@register_loss("unimol_contrastive")
class UniMolLossContrastive(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
    
        
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        
        self.contrastive_loss = NT_Xent(temperature=0.1)

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            encoder_rep
        ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens)
        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        if encoder_coord is not None:
            # real = mask + delta
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data
        
        if self.args.contrastive_loss > 0:
            # contrastive learning
            zis, zjs = encoder_rep.chunk(2)
            contrastive_loss = self.contrastive_loss(zis, zjs)
            loss += contrastive_loss * self.args.contrastive_loss
            logging_output["contrastive_loss"] = contrastive_loss.data
            

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )
        metrics.log_scalar("contrastive_loss", sum(log.get("contrastive_loss", 0) for log in logging_outputs) / sample_size, sample_size, round=6)

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample["net_input"]["src_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        #
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


@register_loss("unimol_infer_contrastive")
class UniMolInferLossContrastive(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        target_key = "target"
        src_tokens = sample[input_key]["src_tokens"].ne(self.padding_idx)
        (
            encoder_rep,
            encoder_pair_rep,
        ) = model(**sample[input_key], features_only=True)
        sample_size = sample[input_key]["src_tokens"].size(0)
        encoder_pair_rep_list = []
        if 'pdb_id' in sample[target_key].keys():
            name_key = 'pdb_id'
        elif 'smi_name' in sample[target_key].keys():
            name_key = 'smi_name'
        else:
            raise NotImplementedError("No name key in the original data")

        for i in range(sample_size):  # rm padding token
            encoder_pair_rep_list.append(encoder_pair_rep[i][src_tokens[i], :][:, src_tokens[i]].data.cpu().numpy())
        logging_output = {
                "mol_repr_cls": encoder_rep[:, 0, :].data.cpu().numpy(),  # get cls token
                "pair_repr": encoder_pair_rep_list,
                "data_name": sample[target_key][name_key],
                "bsz": sample[input_key]["src_tokens"].size(0),
            }
        return 0, sample_size, logging_output

