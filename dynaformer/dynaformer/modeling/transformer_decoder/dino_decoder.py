# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Tan-Cong Nguyen.
# ------------------------------------------------------------------------

from typing import Optional, List, Union
import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast
import torch.nn.functional as F
import math

from ...utils.utils import MLP, _get_clones, _get_activation_fn, inverse_sigmoid,gen_sineembed_for_position,sineembed_for_position_xy,get_bounding_boxes
from ..pixel_decoder.ops.modules import MSDeformAttnMask
from .light_maskcnn_encoder import LightMaskEncoder
from .sinembed_mask_encoder import get_sinusoidal_embedding, gen_sineembed_for_mask

class TransformerDecoder(nn.Module):

    def __init__(self, 
                decoder_layer,
                num_layers,
                norm=None,
                return_intermediate=False,
                d_model=256, 
                query_dim=4,
                modulate_hw_attn=True,
                num_feature_levels=1,
                deformable_decoder=True,
                decoder_query_perturber=None,
                dec_layer_number=None,  # number of queries each layer in decoder
                rm_dec_query_scale=True,
                dec_layer_share=False,
                dec_layer_dropout_prob=None,
                type_mask_embed="MaskSimpleCNN",
                binary_semantic_segmenation=False,
                mask_embed_spatial_shape_level=None
                ):
        super().__init__()
        self.binary_semantic_segmenation=binary_semantic_segmenation
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.type_mask_embed=type_mask_embed
        self.mask_embed_spatial_shape_level=mask_embed_spatial_shape_level
        self.ref_mask_head = MLP(2 * d_model, d_model, d_model, 2)
        if self.type_mask_embed == "MaskSimpleCNN":
          self.maskencoder=LightMaskEncoder()
        elif self.type_mask_embed == "SumSinusoidalMask":
          None
        else:
          raise NotImplementedError(f'This method is not implemented yet:{type_mask_embed}')


        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.mask_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        #self.threshold_mask_layer=[0.5 * (layer_id / (self.num_layers-1)) for layer_id in torch.arange(self.num_layers)]     
        self.threshold_mask_layer=torch.cat((torch.linspace(0.3, 0.5, int(2 * self.num_layers / 3)), torch.full((self.num_layers - int(2 * self.num_layers / 3),), 0.5)))   
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttnMask):
                m._reset_parameters()


    def forward(self, tgt,                                                                # (D+Q)*N*C                             
                memory,                                                                   # Sum{WH}*N*C
                mask_features,                                                            # N*C*W*H   
                tgt_mask: Optional[Tensor] = None,                                        # (D+Q)*(D+Q)  
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,                         # N*Sum{WH}
                pos: Optional[Tensor] = None,
                refbboxs_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 4        # (D+Q)*N*4         unsigmoid
                refmasks_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, H * W    # (D+Q)*N*H*W       unsigmoid
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels                 # Level
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2             # Level*2
                valid_ratios: Optional[Tensor] = None,                                    # N*Level*2
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refmasks_unsigmoid: nq, bs, 2/4/H,W
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        device = tgt.device

        intermediate = []
        reference_masks = refmasks_unsigmoid.to(device)                 #unsigmoid
        reference_bboxs = refbboxs_unsigmoid.to(device)                 #unsigmoid
        ref_bboxs = [reference_bboxs]
        ref_masks = [reference_masks]
        
        if self.type_mask_embed == "SumSinusoidalMask":
          postion_matrix_embed=get_sinusoidal_embedding(reference_masks.shape[2:],reference_masks.device)

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_bboxs = self.decoder_query_perturber(reference_bboxs)
                reference_masks = self.decoder_query_perturber(reference_masks)
            reference_bboxs_input=reference_bboxs     #unsig
            reference_masks_input=reference_masks     #unsig
            
            thres_mask= reference_masks.sigmoid()
            scale_shape= None if self.mask_embed_spatial_shape_level is None else spatial_shapes[self.mask_embed_spatial_shape_level]
            if self.type_mask_embed == "MaskSimpleCNN":
              query_mask_embed = self.maskencoder(thres_mask,scale_shape) 
            elif self.type_mask_embed == "SumSinusoidalMask":
              query_mask_embed = gen_sineembed_for_mask((thres_mask>0.5)*1.0,postion_matrix_embed,scale_shape) 
            else:
              raise NotImplementedError(f'This method is not implemented yet:{self.type_mask_embed}')
            
            if self.binary_semantic_segmenation == True:
              query_centermask_embed = sineembed_for_position_xy(get_bounding_boxes(reference_masks>0)[:,:,:2])
            else:
              query_centermask_embed = sineembed_for_position_xy(reference_bboxs.sigmoid()[:,:,:2])
            
            query_sine_embed=torch.cat((query_mask_embed, query_centermask_embed), dim=-1)
            raw_query_mask = self.ref_mask_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_mask = pos_scale * raw_query_mask

            output = layer(
                tgt=output,                                               #(D+Q)*N*C
                tgt_query_mask=query_mask,                                #(D+Q)*N*C
                tgt_query_sine_embed=query_sine_embed,                    #(D+Q)*N*2C
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_bboxs=reference_bboxs_input,               #(D+Q)*N*2             unsig                
                tgt_reference_masks=reference_masks_input,               #(D+Q)*N*H*W           unsig
                mask_threshold=self.threshold_mask_layer[layer_id],

                memory=memory,                                            #Sum{WH}*N*C
                memory_key_padding_mask=memory_key_padding_mask,          #N*Sum{WH}
                memory_level_start_index=level_start_index,               #Level
                memory_spatial_shapes=spatial_shapes,                     #3*2
                memory_pos=pos,                                           #None

                self_attn_mask=tgt_mask,                                  #(D+Q)*(D+Q)
                cross_attn_mask=memory_mask                               #None
            )
            output_norm=self.norm(output)
            # iter update for bbox
            if self.bbox_embed is not None:
                #(D+Q)*N*4 unsigmoid                        #(D+Q)*N*C                                                                 
                delta_bbox_unsig = self.bbox_embed[layer_id](output_norm).to(device)
                                                        #(D+Q)*N*4
                outputs_bbox_unsig = delta_bbox_unsig + reference_bboxs
                new_reference_bboxs = outputs_bbox_unsig
                reference_bboxs = new_reference_bboxs.detach()
                ref_bboxs.append(new_reference_bboxs)

            # iter update for mask
            if self.mask_embed is not None:
                #unsigmoid                                                                   
                delta_unsig = self.mask_embed[layer_id](output_norm).to(device)
                                                           #N*(D+Q)*C                   #N*C*H*W
                delta_unsig=torch.einsum("bqc,bchw->bqhw", delta_unsig.transpose(0, 1), mask_features) #N*(D+Q)*H*W
                                #(D+Q)*N*H*W                  #(D+Q)*N*H*W
                outputs_unsig = (delta_unsig.transpose(0, 1) + reference_masks)/2
                new_reference_masks = outputs_unsig
                                  #unsigmoid
                reference_masks = new_reference_masks.detach()  #(D+Q)*N*H*W  unSigmoid
                # if layer_id != self.num_layers - 1:
                ref_masks.append(new_reference_masks)         #unSigmoid

            intermediate.append(output_norm)

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],            #list[N*(D+Q)*C]
            [itm_refbbox.transpose(0, 1) for itm_refbbox in ref_bboxs],        #list[N*(D+Q)*4]            #unsigmoid
            [itm_refmask.transpose(0, 1) for itm_refmask in ref_masks]        #list[N*(D+Q)*H*W]          #unsigmoid
        ]


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=8,
                 type_sampling_location="mask",
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MSDeformAttnMask(d_model, n_levels, n_heads, n_points,type_sampling_location)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt



    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model                                         #(D+Q)*N*C                              
                tgt_query_mask: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))         #(D+Q)*N*C
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)        #(D+Q)*N*2C
                tgt_key_padding_mask: Optional[Tensor] = None,                                    #None
                tgt_reference_bboxs: Optional[Tensor] = None,                                     #(D+Q)*N*4   unsig     
                tgt_reference_masks: Optional[Tensor] = None,                                     #(D+Q)*N*H*W unsigmoid
                mask_threshold: Optional[int]=0.5,

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model                               #Sum{WH}*N*C
                memory_key_padding_mask: Optional[Tensor] = None,                                 #N*Sum{WH}
                memory_level_start_index: Optional[Tensor] = None,  # num_levels                  #Level
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2              #3*2
                memory_pos: Optional[Tensor] = None,  # pos for memory                            #None

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention          #(D+Q)*(D+Q)
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention        #None
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_mask)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        #(D+Q)*N*C
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_mask).transpose(0, 1),                    #N*(D+Q)*C
                              tgt_reference_bboxs.transpose(0, 1).contiguous(),                            #N*(D+Q)*4    unsig
                              tgt_reference_masks.transpose(0, 1).contiguous(),                            #N*(D+Q)*H*W  unsig
                              mask_threshold,
                              memory.transpose(0, 1),                                                      #N*Sum{WH}*C 
                              memory_spatial_shapes,                                                       #3*2
                              memory_level_start_index,                                                    #Level
                              #N*Sum{WH}
                              memory_key_padding_mask).transpose(0, 1)                                     
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt   #(D+Q)*N*C   


