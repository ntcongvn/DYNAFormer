# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------
import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks

from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from ...utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid,inverse_sigmoid_mask, apply_random_mask_noise_transforms,get_bounding_boxes_ohw
from ...utils import box_ops
from ...utils import test

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in DYNAFormer.
"""

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.DYNAFormer.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


@TRANSFORMER_DECODER_REGISTRY.register()
class DYNAFormerDecoder(nn.Module):
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
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 8,
            type_sampling_location: str ='both',
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            type_mask_embed: str = 'MaskSimpleCNN'
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
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 4 -> (x, y, w, h)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn=dn
        self.learn_tgt = learn_tgt
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.num_heads = nheads
        self.type_sampling_location=type_sampling_location
        self.num_layers = dec_layers
        self.two_stage=two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_box_embed = nn.Embedding(num_queries, 4)                 
            #self.query_mask_embed = nn.Embedding(num_queries, 224,224)          #Mask in layer scale 1/4
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes=num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        
        #CLASS EMBED
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes+1)
                if num_classes==1:
                  self.binary_semantic_segmenation=True
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc=nn.Embedding(num_classes,hidden_dim)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points, self.type_sampling_location)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          type_mask_embed=type_mask_embed,
                                          binary_semantic_segmenation=self.binary_semantic_segmenation if self.binary_semantic_segmenation is not None else False
                                          )

        self.hidden_dim = hidden_dim

        #Intern class Embed
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.intern_class_embed = nn.Linear(hidden_dim, num_classes+1)
            else:
                self.intern_class_embed = nn.Linear(hidden_dim, num_classes)
        #Intern Box Embed
        self.intern_box_embed =  MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.intern_box_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.intern_box_embed.layers[-1].bias.data, 0)
        #Intern Mask Embed
        self.intern_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        

        #MASK EMBED
        self._mask_embed = _mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        mask_embed_layerlist = [_mask_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.mask_embed = nn.ModuleList(mask_embed_layerlist)
        self.decoder.mask_embed = self.mask_embed

        #BOX EMBED
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels                                                                            #256
        ret["mask_classification"] = mask_classification                                                            #True

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES                                                     #2
        ret["hidden_dim"] = cfg.MODEL.DYNAFormer.HIDDEN_DIM                                                         #256
        ret["num_queries"] = cfg.MODEL.DYNAFormer.NUM_OBJECT_QUERIES                                                #100
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.DYNAFormer.NHEADS                                                                 #8
        ret["dim_feedforward"] = cfg.MODEL.DYNAFormer.DIM_FEEDFORWARD                                               #2048
        ret["dec_layers"] = cfg.MODEL.DYNAFormer.DEC_LAYERS                                                         #9
        ret["dec_n_points"] = cfg.MODEL.DYNAFormer.DEC_N_POINTS
        ret["type_sampling_location"]= cfg.MODEL.DYNAFormer.TYPE_SAMPLING_LOCATIONS
        ret["enforce_input_project"] = cfg.MODEL.DYNAFormer.ENFORCE_INPUT_PROJ                                      #False
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM                                                           #256
        ret["two_stage"] =cfg.MODEL.DYNAFormer.TWO_STAGE                                                            #True
        ret["initialize_box_type"] = cfg.MODEL.DYNAFormer.INITIALIZE_BOX_TYPE  # ['no', 'bitmask', 'mask2box']      #"bitmask"
        ret["dn"]=cfg.MODEL.DYNAFormer.DN                                                                           #"seg"
        ret["noise_scale"] =cfg.MODEL.DYNAFormer.DN_NOISE_SCALE                                                     #0.4
        ret["dn_num"] =cfg.MODEL.DYNAFormer.DN_NUM                                                                  #100
        ret["initial_pred"] =cfg.MODEL.DYNAFormer.INITIAL_PRED                                                      #True
        ret["learn_tgt"] = cfg.MODEL.DYNAFormer.LEARN_TGT                                                           #False
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS                           #3
        ret["semantic_ce_loss"] = cfg.MODEL.DYNAFormer.TEST.SEMANTIC_ON and cfg.MODEL.DYNAFormer.SEMANTIC_CE_LOSS and ~cfg.MODEL.DYNAFormer.TEST.PANOPTIC_ON
        #                                                   False                                False                                           False
        ret["type_mask_embed"] = cfg.MODEL.DYNAFormer.TYPE_MASK_EMBED
        return ret

    def prepare_for_dn(self, targets, tgt, refbox_emb, refmask_emb, batch_size,new_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refmask_emb: mask anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale        #100      0.4

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                input_query_mask = None
                attn_mask = None
                mask_dict = None
                return input_query_label,input_query_bbox, input_query_mask, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_mask = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])        #NxO
            boxes = torch.cat([t['boxes'] for t in targets])          #NxO*4         
            masks = torch.cat([t['masks'] for t in targets])          #NxO*H*W
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

            # known
            known_indice = torch.nonzero(unmask_label + unmask_mask)
            known_indice = known_indice.view(-1)
            
            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)    #scalarxNxO
            known_labels = labels.repeat(scalar, 1).view(-1)          #scalarxNxO
            known_boxes = boxes.repeat(scalar, 1)                     #scalarxNxO*4
            known_bid = batch_idx.repeat(scalar, 1).view(-1)          #scalarxNxO
            known_masks = masks.repeat(scalar, 1,1)                   #scalarxNxO*H*W                  
            known_labels_expaned = known_labels.clone()               #scalarxNxO
            known_boxes_expaned = known_boxes.clone()                 #scalarxNxO*4  Sigmoid
            known_masks_expand = known_masks.clone()                  #scalarxNxO*H*W

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                #Class
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
                #Mask
                known_masks_expand=apply_random_mask_noise_transforms(known_masks_expand, known_boxes, noise_scale,new_size)
                #Box
                known_boxes_expaned=get_bounding_boxes_ohw(known_masks_expand>0) 

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_box_embed = inverse_sigmoid(known_boxes_expaned)
            input_mask_embed = inverse_sigmoid_mask(known_masks_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()
            padding_mask = torch.zeros(pad_size, input_mask_embed.shape[-2],input_mask_embed.shape[-1],dtype=torch.float16).cuda()

            if (not refmask_emb is None) and (not refbox_emb is None):
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refbox_emb], dim=0).repeat(batch_size, 1, 1)
                input_query_mask = torch.cat([padding_mask, refmask_emb], dim=0).repeat(batch_size, 1, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
                input_query_mask = padding_mask.repeat(batch_size, 1, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_box_embed
                input_query_mask[(known_bid.long(), map_known_indice)] = input_mask_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_masks': (known_labels, known_masks),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if (not refmask_emb is None) and (not refbox_emb is None):
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refbox_emb.repeat(batch_size, 1, 1)
                input_query_mask = refmask_emb.repeat(batch_size, 1, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
                input_query_mask=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_mask is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox
            input_query_mask = input_query_mask
                                 #unsigmoid
        return input_query_label,input_query_bbox,input_query_mask,attn_mask,mask_dict

    def dn_post_process(self,outputs_class,   #L*N*(D+Q)*2
                              outputs_bbox,   #L*N*(D+Q)*4
                              outputs_mask,   #L*N*(D+Q)*H*W
                              mask_dict):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_bbox = outputs_bbox[:, :, :mask_dict['pad_size'], :]
        outputs_bbox = outputs_bbox[:, :, mask_dict['pad_size']:, :]
        output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
        outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        
        out = {'pred_logits': output_known_class[-1],'pred_boxes': output_known_bbox[-1], 'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_bbox)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class,outputs_bbox, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    

    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        if self.two_stage:
            (H_max,W_max)=spatial_shapes[0]
                            #unsig
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
  
            #Class unselected
            enc_outputs_class_unselected = self.intern_class_embed(output_memory)
            enc_outputs_coord_unselected = self.intern_box_embed(output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid

            topk = self.num_queries

            if self.binary_semantic_segmenation is not None and self.binary_semantic_segmenation== True:
              topk_proposals = torch.topk(enc_outputs_class_unselected[...,0], topk, dim=1)[1]
            else:  
              #instance segm
              topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

            #Memory
            #N*HW*256
            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # (bs, \sum{hw}, C)

            #Intern Class
            interm_outputs_class=self.intern_class_embed(tgt_undetach)                                  #logits
            #Intern Box
            interm_outputs_box=torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))        # unsigmoid
            #interm_outputs_box=self.intern_box_embed(tgt_undetach)                                     #unsig
            #Intern Mask
            intern_mask_embed=self.intern_mask_embed(tgt_undetach)
            interm_outputs_mask=torch.einsum("bqc,bchw->bqhw", intern_mask_embed, mask_features)        #unsig
            

            interm_outputs=dict()
            interm_outputs['pred_logits'] = interm_outputs_class                                        # N*Q*2    logits
            interm_outputs['pred_boxes'] = interm_outputs_box.sigmoid()                                 # N*Q*4    sigmoid
            interm_outputs['pred_masks'] = interm_outputs_mask                                          # N*Q*H*W  unsigmoid
            
            # Detach for decoder processing
            tgt = tgt_undetach.detach()
            refbbox_embed = interm_outputs_box.detach()       #unsig
            refmask_embed = interm_outputs_mask.detach()      #unsig

            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)

            #We use refmask_embed insteal of refbox_embed, but initialize box is better  
            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = refmask_embed.flatten(0, 1)
                h, w = refmask_embed.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    refbbox_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    refbbox_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refbbox_embed = box_ops.box_xyxy_to_cxcywh(refbbox_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refbbox_embed = refbbox_embed.reshape(refmask_embed.shape[0], refmask_embed.shape[1], 4)
                refbbox_embed = inverse_sigmoid(refbbox_embed)

        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refbbox_embed = self.query_box_embed.weight[None].repeat(bs, 1, 1)
            refmask_embed=self.decoder_norm(tgt.transpose(0, 1)).transpose(0, 1)
            refmask_embed=torch.einsum("bqc,bchw->bqhw", refmask_embed, mask_features)
            #refmask_embed = self.query_mask_embed.weight[None].repeat(bs, 1, 1).view(bs,-1, 224, 224)
        
        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
                                #unsigmoid
            input_query_label,input_query_bbox, input_query_mask, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None,None, x[0].shape[0],refmask_embed.shape[-2:])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        if self.dn != "no" and self.training and mask_dict is not None:
            #unsig                   #unsig           #unsig
            refmask_embed=torch.cat([input_query_mask,refmask_embed],dim=1)
            refbbox_embed=torch.cat([input_query_bbox,refbbox_embed],dim=1)

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            #N*(D+Q)*C                              #N*(D+Q)*C
            outputs_class = self.forward_prediction_class_heads(tgt)
            predictions_class.append(outputs_class)               #logits
        
            #unsigmoid        #unsigmoid
        hs, references_bbox, references_mask = self.decoder(
            tgt=tgt.transpose(0, 1),                              # (D+Q)*N*C
            memory=src_flatten.transpose(0, 1),                   # Sum{WH}*N*C
            mask_features=mask_features,                          # N*C*W*H                          
            memory_key_padding_mask=mask_flatten,                 # N*Sum{WH}
            pos=None,
            refbboxs_unsigmoid=refbbox_embed.transpose(0, 1),     # (D+Q)*N*H*W           unsig
            refmasks_unsigmoid=refmask_embed.transpose(0, 1),     # (D+Q)*N*H*W           unsig
            level_start_index=level_start_index,                  # Level
            spatial_shapes=spatial_shapes,                        # Level*2
            valid_ratios=valid_ratios,                            # N*Level*2
            tgt_mask=tgt_mask                                     # (D+Q)*(D+Q)
        )
        
        # iteratively class and box  prediction
        for i, output in enumerate(hs):
            #hs is already normalized, we can predict directly
            outputs_class=self.class_embed(output)
            predictions_class.append(outputs_class)                                                     #logits

        if self.initial_pred:
            #sig              #unsigmoid
            predictions_box,  predictions_mask  = self.forward_prediction_bbox_and_mask_heads(references_bbox,      #list[N*(D+Q)*4]    unsign
                              references_mask,                                                                      #list[N*(D+Q)*H*W]    unsign
                              hs,                                                                                   #list[N*(D+Q)*C]
                              mask_features,                                                                        #N*C*W*H
                              refbbox_embed,                                                                        #N*(D+Q)*4            unsig
                              refmask_embed)                                                                        #N*(D+Q)*H*W          unsig     
        else:
            predictions_box,  predictions_mask  = self.forward_prediction_bbox_and_mask_heads(references_bbox,
                                                                                            references_mask,
                                                                                            hs,
                                                                                            mask_features)

        assert len(predictions_class) == self.num_layers + 1 and len(predictions_class)==len(predictions_box) and len(predictions_box)==len(predictions_mask)
        
        if mask_dict is not None:
            predictions_class=torch.stack(predictions_class)                                      #L*N*(D+Q)*2
                                                #unsigmoid
            predictions_class,predictions_box,predictions_mask=self.dn_post_process(predictions_class,         #L*N*(D+Q)*2
                                                              predictions_box,           #L*N*(D+Q)*4
                                                              predictions_mask,          #L*N*(D+Q)*H*W
                                                              mask_dict)          
                                                                                   
            predictions_class=list(predictions_class)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()
        
        out = {
            'pred_logits': predictions_class[-1],                 #logits
            'pred_boxes':predictions_box[-1],
            'pred_masks': predictions_mask[-1],                   #unsigmoid
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None,predictions_mask,predictions_box
            )
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        return out, mask_dict


    def forward_prediction_bbox_and_mask_heads(self,reference_bbox, #list[N*(D+Q)*4]              #unsig
                                                    reference_mask, #list[N*(D+Q)*H*W]            #unsig
                                                    hs,             #list[N*(D+Q)*C]
                                                    mask_features,  #N*C*W*H
                                                    ref_bbox0=None, #N*(D+Q)*4                    #unsig
                                                    ref_mask0=None  #N*(D+Q)*H*W                  #unsig
                                                    ):     
                                                         
        '''
          :param reference: reference mask from each decoder layer
          :param hs: content
          :param ref0: whether there are prediction from the first layer
        '''
        device = reference_mask[0].device

        if ref_bbox0 is None or ref_mask0 is None:
            outputs_bbox_list = []
            outputs_mask_list = []
        else:
            outputs_bbox_list = [ref_bbox0.to(device).sigmoid()]
            outputs_mask_list = [ref_mask0.to(device)]

        for dec_lid, (layer_hs,layer_refbbox_unsig,layer_bbox_embed,layer_refmask_unsig, layer_mask_embed ) in enumerate(zip(hs,reference_bbox[:-1],self.bbox_embed,reference_mask[1:], self.mask_embed)):
            layer_delta_bbox_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_bbox_unsig = layer_delta_bbox_unsig + layer_refbbox_unsig.to(device)
            outputs_bbox_list.append(layer_outputs_bbox_unsig.sigmoid())
            
            mask_embed = layer_mask_embed(layer_hs).to(device)
            layer_delta_mask_unsig=torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)                
            layer_outputs_mask_unsig = (layer_delta_mask_unsig + layer_refmask_unsig.to(device))/2
            outputs_mask_list.append(layer_outputs_mask_unsig.to(device))

        outputs_bbox_list = torch.stack(outputs_bbox_list)
        outputs_mask_list = torch.stack(outputs_mask_list)
               #sigmoid           #unsigmoid
        return outputs_bbox_list, outputs_mask_list                      

    def forward_prediction_class_heads(self, output):
        decoder_output = self.decoder_norm(output.transpose(0, 1))
        outputs_class = self.class_embed(decoder_output.transpose(0, 1))
        return outputs_class 

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]