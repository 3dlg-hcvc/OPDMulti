# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .mask2former_transformer_decoder import (
    SelfAttentionLayer,
    CrossAttentionLayer,
    FFNLayer,
    MLP,
)
from ..criterion import convert_to_filled_tensor


@TRANSFORMER_DECODER_REGISTRY.register()
class OPDMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # OPD
        motionnet_type,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # OPD
        self.motionnet_type = motionnet_type
        self.num_classes = num_classes

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes + 1),
            )
            # OPD Changes
            self.mtype_embed = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 2),
            )
            self.morigin_embed = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 3),
            )
            self.maxis_embed = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 3),
            )
            if self.motionnet_type == "BMOC_V0":
                # Define the layers for the extrinsic prediction
                self.extrinsic_feature_layer = nn.Sequential(
                    # 16 * 256 * 64 * 64
                    nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 32 * 32
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),  
                    nn.MaxPool2d(2, 2), # 16 * 256 * 16 * 16
                    nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 8 * 8
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),  
                    nn.MaxPool2d(2, 2), # 16 * 256 * 4 * 4
                    nn.Conv2d(256, 64, 1), # 16 * 64 * 4 * 4
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),  
                    nn.Flatten() # 16 * 1024
                )
                for layer in self.extrinsic_feature_layer:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            layer.weight, mode="fan_out", nonlinearity="relu"
                        )
                self.extrinsic_pred_layer = nn.Sequential(
                    nn.Linear(768, 512),
                    # nn.Linear(768, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 12), # 16 * 12
                )
            elif self.motionnet_type == "BMOC_V1":
                self.extrinsic_embed = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 12),
                )
            elif self.motionnet_type == "BMOC_V2":
                self.extrinsic_embed = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 7),
                )
            elif self.motionnet_type == "BMOC_V3":
                self.extrinsic_embed = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 9),
                )
            elif self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
                if self.motionnet_type == "BMOC_V5":
                    self.mask_weight_layer = SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                # Define the layers for the extrinsic prediction
                self.extrinsic_feature_layer = nn.Sequential(
                    nn.BatchNorm2d(256),
                    # 16 * 256 * 64 * 64
                    nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 32 * 32
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),  
                    nn.MaxPool2d(2, 2), # 16 * 256 * 16 * 16
                    nn.Conv2d(256, 256, 3, 2, 1), # 16 * 256 * 8 * 8
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),  
                    nn.MaxPool2d(2, 2), # 16 * 256 * 4 * 4
                    nn.Conv2d(256, 64, 1), # 16 * 64 * 4 * 4
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),  
                    nn.Flatten() # 16 * 1024
                )
                for layer in self.extrinsic_feature_layer:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            layer.weight, mode="fan_out", nonlinearity="relu"
                        )
                if self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5":
                    self.extrinsic_pred_layer = nn.Sequential(
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 32),
                        nn.ReLU(inplace=True),
                        nn.Linear(32, 7), # 16 * 7
                    )
                elif self.motionnet_type == "BMOC_V6":
                    self.extrinsic_pred_layer = nn.Sequential(
                        # nn.Linear(1024, 512),
                        nn.Linear(768, 512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 32),
                        nn.ReLU(inplace=True),
                        nn.Linear(32, 12), # 16 * 12
                    )
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        
        # OPD
        ret["motionnet_type"] = cfg.MODEL.MOTIONNET.TYPE

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        # OPD
        predictions_mtype = []
        predictions_morigin = []
        predictions_maxis = []

        if self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2" or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
            predictions_extrinsic = []


        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_mtype, outputs_morigin, outputs_maxis, outputs_extrinsic = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], query_embed=query_embed)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        # OPD
        predictions_mtype.append(outputs_mtype)
        predictions_morigin.append(outputs_morigin)
        predictions_maxis.append(outputs_maxis)

        if self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2" or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
            predictions_extrinsic.append(outputs_extrinsic)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, outputs_mtype, outputs_morigin, outputs_maxis, outputs_extrinsic = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], query_embed=query_embed)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            # OPD
            predictions_mtype.append(outputs_mtype)
            predictions_morigin.append(outputs_morigin)
            predictions_maxis.append(outputs_maxis)

            if self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2" or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
                predictions_extrinsic.append(outputs_extrinsic)

        assert len(predictions_class) == self.num_layers + 1
        if self.mask_classification:
            if self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMCC":
                aux_outputs = self._set_aux_loss(
                        predictions_class, predictions_mask, predictions_mtype, predictions_morigin, predictions_maxis, None
                    )
            elif self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2"  or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
                aux_outputs = self._set_aux_loss(
                        predictions_class, predictions_mask, predictions_mtype, predictions_morigin, predictions_maxis, predictions_extrinsic
                    )
            
        else:
            aux_outputs = self._set_aux_loss(
                    None, predictions_mask, None, None, None, None
                )
        # OPD
        if self.motionnet_type == "BMOC_V0":
            extrinsic_feature = self.extrinsic_feature_layer(mask_features)
            predictions_extrinsic = self.extrinsic_pred_layer(extrinsic_feature)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            # OPD
            'pred_mtypes': predictions_mtype[-1],
            'pred_morigins': predictions_morigin[-1],
            'pred_maxises': predictions_maxis[-1],
            'aux_outputs': aux_outputs
        }
        if self.motionnet_type == "BMOC_V0":
            out['pred_extrinsics'] = predictions_extrinsic
        elif self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2"  or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
            out['pred_extrinsics'] = predictions_extrinsic[-1]

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, query_embed):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        # OPD Changes
        outputs_mtype = self.mtype_embed(decoder_output)
        outputs_morigin = self.morigin_embed(decoder_output)
        outputs_maxis = self.maxis_embed(decoder_output)

        if self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2"  or self.motionnet_type == "BMOC_V3":
            outputs_extrinsic = self.extrinsic_embed(decoder_output)
        elif self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMCC":
            outputs_extrinsic = None

        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V6":
            binary_mask = (outputs_mask > 0).float()
            weighted_masked_feature = mask_features + torch.einsum("bqhw,bchw->bchw", binary_mask, mask_features)
            extrinsic_feature = self.extrinsic_feature_layer(weighted_masked_feature)
            outputs_extrinsic = self.extrinsic_pred_layer(extrinsic_feature)
        elif  self.motionnet_type == "BMOC_V5":
            # Get one weight for each query
            mask_weights = torch.transpose(self.mask_weight_layer(
                torch.transpose(mask_embed, 0, 1), tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            ), 0, 1).mean(2)
            binary_mask = (outputs_mask > 0).float()
            weighted_mask = torch.einsum("bq,bqhw->bqhw", mask_weights, binary_mask)
            weighted_masked_feature = mask_features + torch.einsum("bqhw,bchw->bchw", weighted_mask, mask_features)
            extrinsic_feature = self.extrinsic_feature_layer(weighted_masked_feature)
            outputs_extrinsic = self.extrinsic_pred_layer(extrinsic_feature)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, outputs_mtype, outputs_morigin, outputs_maxis, outputs_extrinsic

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, predictions_mtype, predictions_morigin, predictions_maxis, predictions_extrinsic):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            if self.motionnet_type == "BMOC_V0" or self.motionnet_type == "BMCC":
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_mtypes": c, "pred_morigins": d, "pred_maxises": e}
                    for a, b, c, d, e in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_mtype[:-1], predictions_morigin[:-1], predictions_maxis[:-1])
                ]
            elif self.motionnet_type == "BMOC_V1" or self.motionnet_type == "BMOC_V2"  or self.motionnet_type == "BMOC_V3" or self.motionnet_type == "BMOC_V4" or self.motionnet_type == "BMOC_V5" or self.motionnet_type == "BMOC_V6":
                return [
                    {"pred_logits": a, "pred_masks": b, "pred_mtypes": c, "pred_morigins": d, "pred_maxises": e, "pred_extrinsics": f}
                    for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_mtype[:-1], predictions_morigin[:-1], predictions_maxis[:-1], predictions_extrinsic[:-1])
                ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
