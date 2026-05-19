# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Hugging Face Transformers (https://github.com/huggingface/transformers)
# Copyright (c) The Hugging Face team. All rights reserved.
# ------------------------------------------------------------------------

from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLModelOutputWithPast
import torch

from transformers.utils import  auto_docstring, can_return_tuple

from dataclasses import dataclass
from typing import  List, Optional, Tuple, Union
from torch import nn
from ...datasets.utils.constants import IGNORE_INDEX, POS_INDICATOR_TOKEN_INDEX, POS_EMBEDDING_TOKEN_INDEX


from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import (
    GenerationConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

from transformers.generation.utils import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
from ..vlm_utils.positional_encoding import RoPE3D


import os

@dataclass
class CustomGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    hidden_states_for_output: Optional[torch.FloatTensor] = None  # Original last hidden state before any modifications

@dataclass
class CustomQwen2_5_VLCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    loss_pos: Optional[torch.FloatTensor] = None  # L2 loss for positional embeddings
    output_pos: Optional[torch.FloatTensor] = None  # Output positional embeddings
    gt_pos: Optional[torch.FloatTensor] = None  # Ground truth positional embeddings
    output_pe_mask: Optional[torch.BoolTensor] = None  # Mask for output positional embeddings
    gt_coords_xy: Optional[torch.FloatTensor] = None  # Ground truth coordinates in xy plane
    last_hidden_state: Optional[torch.FloatTensor] = None  # Last hidden state of the model
    last_hidden_state_original: Optional[torch.FloatTensor] = None  # Original last hidden state before any modifications
    labels: Optional[torch.LongTensor] = None  # Labels for the loss calculation


class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, ):
        super().__init__(config)
        self.model = CustomQwen2_5_VLModel(config)

        self.l2_loss = nn.MSELoss()
  
        self.llm_hidden_dim = config.text_config.hidden_size
        self.position_encoder = None #PositionalEncoding3D(self.llm_hidden_dim, dtype_override=torch.bfloat16,freq_coeff=20000) # input (batch_size, num_pixels, 3) output (batch_size, num_pixels, llm_hidden_dim)
        self.pc_range = None # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] # NOTE: find a way to load train_cfg['point_cloud_range']
        self.voxel_size = None #  [0.4, 0.4, 8]  #[0.2, 0.2, 8] # NOTE: this is the voxel size for the grid 32.00 GiB for [0.05, 0.05, 8]
        self.pos_emb_grid = None # self.position_encoder.pos_grid_3d(self.pc_range, voxel_size=self.voxel_size)




    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        io_coords_pos: Optional[torch.Tensor] = None,
        loss_pos_lambda: Optional[torch.FloatTensor] = None,
        loss_for_pos: str ='l2',
        include_semantic_posemb = False,
        supervise_semantic_posemb = False,
        planning_only: Optional[bool] = False,
        single_coords_only: Optional[bool] = False,
        has_gt_planning = torch.zeros(1),
        gt_coords_xy: Optional[torch.Tensor] = None,
        coords_encoder = None,
        coords_decoder = None, # NOTE: this is only for test inference
        ego_feature: Optional[torch.Tensor] = None, # shape (B, 1, hidden)
        enable_pe_input = False, # enable the use of PE in autoregressive manner, if False, only use PE for supervision
        pos_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        if pixel_values is not None:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
            image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # NOTE: 1. find all POS_EMBEDDING_TOKEN_INDEX in input_ids, create a io_coords_pos_mask
        io_coords_pos_mask= None
        weighted_mask = None
        if io_coords_pos is not None:
            weighted_tokens = [
                POS_INDICATOR_TOKEN_INDEX,
            ]
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[weighted_tokens] = 3.0
            weighted_mask= weighted_mask.float()

            # convert io_coords_pos to the same dtype as model parameters
            io_coords_pos = io_coords_pos.to(dtype=self.model.dtype, device=self.model.device)

            
            # create a mask for the positions where we have io_coords_pos
            io_coords_pos_mask = (input_ids == POS_EMBEDDING_TOKEN_INDEX).unsqueeze(-1)

            # NOTE: 1.1. io_coords_pos has shape (bs, num_coords, pos_dim) now. It should be (bs, len_input_ids, pos_dim). Position that is not coords should be filled with 0
            io_coords_pos_indices = io_coords_pos_mask.nonzero(as_tuple=False)
            scattered_io_coords_pos = torch.full(
                (input_ids.shape[0], input_ids.shape[1], io_coords_pos.shape[-1]),
                0,
                dtype=io_coords_pos.dtype,
                device=io_coords_pos.device,
            )
            if io_coords_pos_indices.shape[0] == io_coords_pos.shape[1] :
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos.reshape(-1,io_coords_pos.shape[-1] )
                io_coords_pos = scattered_io_coords_pos.reshape(B , -1, io_coords_pos.shape[-1]) # now has shape (bs, len_input_ids, pos_dim)
            else:
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos[:, :io_coords_pos_indices.shape[0]].reshape(-1,io_coords_pos.shape[-1] )  # only use the first num_coords positions
                io_coords_pos = scattered_io_coords_pos.reshape(B , -1, io_coords_pos.shape[-1])



        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            pos_emb=pos_emb,
            io_coords_pos_mask=io_coords_pos_mask, # This is a 0,1 mask
            enable_pe_input = enable_pe_input, # enable the use of PE in autoregressive manner, if False, only use PE for supervision
            io_coords_pos = io_coords_pos,
            include_semantic_posemb = include_semantic_posemb,
            ego_feature = ego_feature,
            pos_index = pos_index,
        )

        hidden_states = outputs[0] # Note: This is hidden states before the cls head
        hidden_states_original = hidden_states # save the original hidden states for later use
        if hidden_states.dtype != self.lm_head.weight.dtype:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)

        logits = self.lm_head(hidden_states)
        if hasattr(self, 'new_lm_head'):
            if self.new_lm_head.weight.dtype != logits.dtype:
                hidden_states = hidden_states.to(self.new_lm_head.weight.dtype)
            new_logits = self.new_lm_head(hidden_states) # shape (bs, seq_len, new_tokens)
            new_logits = new_logits.to(logits.dtype)

            # replace the logits for new tokens
            logits = logits.index_copy(2, torch.arange(151665, 151667, device=logits.device), new_logits)



        # NOTE: (4. Training only) use io_coords_pos_mask to select pos out of hidden_states
        if io_coords_pos_mask is not None:
            # first create a mask to only select pos in answers, not in input.
            # which is position that are not IGNORE_INDEX
            output_mask = (labels != IGNORE_INDEX).unsqueeze(-1).expand_as(hidden_states)
            output_pe_mask = torch.roll(io_coords_pos_mask, shifts=-1, dims=-2) # roll one to left, output_pe_mask has shape (bs, len_input_ids, 2048) # this is the mask for the positions where we have io_coords_pos
            output_mask_hidden = torch.roll(output_mask, shifts=-1, dims=-2) # roll one to left,  has shape (bs, len_input_ids, 2048)
            output_pe_mask = output_pe_mask & output_mask_hidden # should be (bs, len_input_ids, hidden_states)
            output_pos = hidden_states.masked_select(output_pe_mask).view(-1, hidden_states.shape[-1])
            if include_semantic_posemb:
                # need to reduce features of POS_EMBEDDING_TOKEN_INDEX from output_pos
                semantic_pos_emb = self.model.get_input_embeddings()(torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=io_coords_pos.device).unsqueeze(0)).squeeze(0) # semantic_pos_emb is the feature of the POS_EMBEDDING_TOKEN_INDEX token
                output_pos = output_pos - semantic_pos_emb # reduce the semantic pos emb from the output_poss

            
            # select the ground truth pos from io_coords_pos
            gt_pos = io_coords_pos.masked_select(io_coords_pos_mask & output_mask).view(-1, io_coords_pos.shape[-1])

            if gt_coords_xy != None: 
                output_xy_mask = output_mask.masked_select(io_coords_pos_mask).view( -1, hidden_states.shape[-1]) # shape (bs, num_pos, hidden_states)
                output_xy_mask = output_xy_mask[:,:2]
                
                gt_coords_xy = gt_coords_xy.masked_select(output_xy_mask).to(gt_pos.dtype) # shape

        loss = None
        if labels is not None:
            # NOTE: 3. according to io_coords_pos_mask, replace it with -100
            if io_coords_pos_mask is not None:
                if not (supervise_semantic_posemb and include_semantic_posemb):
                    labels = labels.masked_fill(io_coords_pos_mask.squeeze(-1), IGNORE_INDEX)
                else:
                    pass

            if planning_only and single_coords_only:
                raise ValueError("planning_only and single_coords_only cannot be both True at the same time.")
            
            if not has_gt_planning.any():
                output_pos = torch.empty((0, self.llm_hidden_dim)).to(labels.device)
                gt_pos = torch.empty((0,  self.llm_hidden_dim)).to(labels.device)
                gt_coords_xy = torch.empty((0, 2)).to(labels.device)
            else:

                num_valid = has_gt_planning.sum()
                
                if planning_only:
                    # only select the first six coords for loss_pos calculation
                    if io_coords_pos is not None:
                        output_pos = output_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :6, :] # select the first six coords
                        gt_pos = gt_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :6, :] # select the first six coords
                        gt_coords_xy = gt_coords_xy.reshape(num_valid, -1, 2)[:, :6, :] # select the first six coords
                elif single_coords_only:
                    # only select the first coords for loss_pos calculation
                    if io_coords_pos is not None: # NOTE: make sure it also has at least one coords
                        output_pos = output_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :1, :].unsqueeze(1) # select the first coords
                        gt_pos = gt_pos.reshape(num_valid, -1, self.llm_hidden_dim)[:, :1, :].unsqueeze(1) # select the first coords
                        gt_coords_xy = gt_coords_xy.reshape(num_valid, -1, 2)[:, :6, :] # select the first six coords

            
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size , weight=weighted_mask, num_items_in_batch=torch.tensor(B))

            if io_coords_pos_mask is not None:

                if loss_for_pos == 'cosine':
                    loss_pos = 1 - torch.cosine_similarity(output_pos, gt_pos, dim=-1).mean()
                elif loss_for_pos == 'l2':
                    loss_pos = self.l2_loss(output_pos, gt_pos)
                elif loss_for_pos == 'l2_coords':
                    decoded_output_pos, interpolated_output_pos = self.position_encoder.decode_pos( output_pos.reshape(B, -1, self.llm_hidden_dim), self.pos_emb_grid,
                                                                                self.pc_range, self.voxel_size, sim_method='cosine')
                    torch.cuda.empty_cache()
                    decoded_gt_pos, interpolated_gt_pos = self.position_encoder.decode_pos(gt_pos.reshape(B, -1, self.llm_hidden_dim), self.pos_emb_grid,
                                                                            self.pc_range, self.voxel_size, sim_method='cosine')
                    torch.cuda.empty_cache()
                    loss_pos = self.l2_loss(interpolated_output_pos[:,:,:2], interpolated_gt_pos[:,:,:2])
                elif loss_for_pos == 'l2_coords_full_grid':
                    decoded_output_pos = self.position_encoder.decode_pos_full_grid( output_pos.reshape(B, -1, self.llm_hidden_dim), self.pos_emb_grid,
                                                                                self.pc_range, self.voxel_size, sim_method='cosine')
                    torch.cuda.empty_cache()
                    decoded_gt_pos = self.position_encoder.decode_pos_full_grid(gt_pos.reshape(B, -1, self.llm_hidden_dim), self.pos_emb_grid,
                                                                            self.pc_range, self.voxel_size, sim_method='cosine')
                    torch.cuda.empty_cache()
                    loss_pos = self.l2_loss(decoded_output_pos[:,:,:2], decoded_gt_pos[:,:,:2])
                elif loss_for_pos == 'l2_coords_mlp':
                    # use the mlp_1_coords to decode the output_pos and gt_pos
                    output_pos = output_pos.to(torch.float32) # make sure the output_pos is in the same dtype as mlp_1_coords
                    gt_coords_xy = gt_coords_xy.to(torch.float32)

                    loss_pos = torch.tensor(0.0, device=output_pos.device, dtype=output_pos.dtype)
                elif loss_for_pos == 'l2_coords_mlp_2layer':
                    # use the mlp_1_coords to decode the output_pos and gt_pos
                    output_pos = output_pos.to(torch.float32) # make sure the output_pos is in the same dtype as mlp_1_coords
                    gt_coords_xy = gt_coords_xy.to(torch.float32)

                    loss_pos = torch.tensor(0.0, device=output_pos.device, dtype=output_pos.dtype)
                else:
                    raise ValueError(f"Unknown loss_for_pos: {loss_for_pos}. Use 'cosine' or 'l2'.")
                loss_pos = loss_pos * loss_pos_lambda

        if coords_encoder is not None and coords_decoder is not None:
            hs_dtype = hidden_states.dtype
            
            if include_semantic_posemb:
                # remove the semantic pos emb from the hidden states
                semantic_pos_emb = self.model.get_input_embeddings()(torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=hidden_states.device).unsqueeze(0)).squeeze(0)
                hidden_states = hidden_states - semantic_pos_emb

            hidden_states = hidden_states.to(torch.float32) # make sure the hidden_states is in the same dtype as coords_encoder
            decoded_coords = coords_decoder(hidden_states)[:,:,:3] # only use x,y, shape (bs, len_input_ids, 2)
            if decoded_coords.shape[-1] < 3:
                # append 0 in last dim 
                decoded_coords = torch.cat([decoded_coords, torch.zeros((*decoded_coords.shape[:-1],1), device=decoded_coords.device, dtype=decoded_coords.dtype)], dim=-1)
            decoded_coords[:,:,-1] = 0

            reencoded_pe = coords_encoder(decoded_coords)

            
            hidden_states = reencoded_pe.to(hs_dtype) # convert back to the original dtype
            
            if include_semantic_posemb:
                # add the semantic pos emb back to the decoded coords
                hidden_states = hidden_states + semantic_pos_emb



        if not return_dict:
            output = (logits,) + outputs[1:]
            if io_coords_pos_mask is not None:
                return (loss,loss_pos,) + output if loss is not None else output
            else:
                return (loss,) + output if loss is not None else output

        if labels is not None and io_coords_pos_mask is not None:
            return CustomQwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                loss_pos= loss_pos if io_coords_pos_mask is not None else None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                last_hidden_state=hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
                # extra args
                output_pos=output_pos,
                gt_pos=gt_pos,
                output_pe_mask=output_pe_mask,
                gt_coords_xy=gt_coords_xy,
                labels = labels
            )
        return CustomQwen2_5_VLCausalLMOutputWithPast(  # NOTE: This is used for test Inference
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            last_hidden_state=hidden_states,
            last_hidden_state_original=hidden_states_original
        )
    

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        inputs_embeds = None
        input_embeds_for_output = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True
        

        # NOTE: new args that will be used in the loop
        if_next_token_pos = False
        next_token_pos_countdown = -200

        # NOTE: Here starts the generation loop
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            
            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, inputs_embeds= inputs_embeds, if_next_token_pos = if_next_token_pos, next_token_pos_countdown= next_token_pos_countdown, **model_kwargs)
            model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, inputs_embeds= inputs_embeds, **model_kwargs)
            
            # Manually add 'enable_pe_input' to model_inputs
            model_inputs['enable_pe_input'] = model_kwargs.get('enable_pe_input', False)
            

            if if_next_token_pos and not model_inputs['enable_pe_input']:
                # change the input_ids to POS_EMBEDDING_TOKEN_INDEX
                # NOTE: this is only for the next token position, not for the whole input_ids
                model_inputs["input_ids"] = torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=model_inputs["input_ids"].device).reshape(model_inputs["input_ids"].shape)
            elif if_next_token_pos and model_inputs['enable_pe_input']:
                # load the last PE token embedding to inputs_embeds
                model_inputs["inputs_embeds"] = inputs_embeds[:, -1, :].unsqueeze(1) # shape (batch_size, 1, hidden_size)
                model_inputs["input_ids"] = None

            # NOTE: In the next iteration, make sure to set if_next_token_pos to False after use
            if_next_token_pos = False
            next_token_pos_countdown -= 1 # decrease the countdown by 1, if it is not POS_Indicator token, we will set next_token_pos to False

            # NOTE: this is an added operation
            # if the next_token is POS_Indicator, we will set next_token_pos to True
            if model_inputs["input_ids"] is not None and model_inputs["input_ids"][0, -1] == POS_INDICATOR_TOKEN_INDEX:
                if_next_token_pos = True
                next_token_pos_countdown = 0

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                #NOTE: remove the io_coords_pos from model_inputs to avoid error
                if 'io_coords_pos' in model_inputs:
                    model_inputs.pop('io_coords_pos')
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if return inputs_embeds
            if return_dict_in_generate:
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
                    if inputs_embeds is None:
                        inputs_embeds = outputs.last_hidden_state
                        input_embeds_for_output = outputs.last_hidden_state_original
                    else:
                        inputs_embeds = torch.cat([inputs_embeds, outputs.last_hidden_state], dim=1)
                        input_embeds_for_output = torch.cat([input_embeds_for_output, outputs.last_hidden_state_original], dim=1)

            # if we are using synced_gpus, we can skip the rest of the loop
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)



            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            if if_next_token_pos:
                unfinished_sequences = unfinished_sequences
            else:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores) 
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return CustomGenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=inputs_embeds, # This is used for decoding x,y position
                    hidden_states_for_output=input_embeds_for_output,  # Original last hidden state before any modifications
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        


class CustomQwen2_5_VLModel(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.rope_3d = RoPE3D(config.text_config.hidden_size)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        io_coords_pos_mask: Optional[torch.Tensor] = None,
        io_coords_pos: Optional[torch.Tensor] = None,
        include_semantic_posemb: Optional[bool] = False,
        ego_feature: Optional[torch.Tensor] = None,
        enable_pe_input: Optional[bool] = False,
        pos_index: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2_5_VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2_5_VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        NOTE: The official qwen implementation does not support input_embeds, so we still inputs_ids still needs to be passed 
        """
        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is None:
            if hasattr(self.language_model, 'new_embed_tokens'):
                is_new_token = (input_ids >= 151665)
                with torch.no_grad():
                    original_embeds = self.get_input_embeddings()(input_ids)

                new_token_relative_ids = input_ids[is_new_token] - 151665
                new_embeds = self.language_model.new_embed_tokens(new_token_relative_ids)

                inputs_embeds = original_embeds.clone()
                inputs_embeds[is_new_token] = new_embeds.to(inputs_embeds.dtype)
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            
        if inputs_embeds.dtype != torch.bfloat16: 
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        
        # replace coords here (for coords in input)
        if enable_pe_input and io_coords_pos is not None and io_coords_pos_mask is not None:
            inputs_embeds  = inputs_embeds.masked_scatter(io_coords_pos_mask, io_coords_pos)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features and ego_feature is None:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            if ego_feature is not None and n_image_tokens != n_image_features + ego_feature.shape[1] * ego_feature.shape[0]:
                raise ValueError(
                    f"Image features + ego feature and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features} + ego_feature {ego_feature.shape[1] * ego_feature.shape[0]}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            # NOTE: HERE we add the pos_emb on top of the image embeddings
            if pos_index is not None:
                # NOTE: add rope here
                image_embeds = image_embeds.unsqueeze(0)
                image_embeds = self.rope_3d(image_embeds, pos_index)
                image_embeds = image_embeds.squeeze(0)
            elif pos_emb is not None:
                pos_emb = pos_emb.reshape(image_embeds.shape)
                if pos_emb.shape[0] != image_embeds.shape[0]:
                    raise ValueError(
                        f"Positional embedding shape {pos_emb.shape} does not match image embedding shape {image_embeds.shape}"
                    )
                image_embeds = image_embeds + pos_emb # shape [B, N, C]
                
                

            # NOTE: HERE starts the ego feature insertion
            if ego_feature is not None:
                image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1]) # shape (B, N, C)
                image_embeds = torch.cat([image_embeds, ego_feature], dim=-2)
                image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1]) # shape (B*N, C)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. NOTE @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

