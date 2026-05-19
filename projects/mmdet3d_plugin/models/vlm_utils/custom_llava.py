# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Hugging Face Transformers (https://github.com/huggingface/transformers)
# Copyright (c) The Hugging Face team. All rights reserved.
# ------------------------------------------------------------------------

from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration,  LlavaModel, LlavaCausalLMOutputWithPast, KwargsForCausalLM, LlavaModelOutputWithPast, Unpack, is_torchdynamo_compiling
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
import torch

from transformers.utils import  auto_docstring, can_return_tuple

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
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
# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
from ..vlm_utils.positional_encoding import PositionalEncoding3D

import os


@dataclass
class CustomLlavaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    """
    Base class for Llava causal language model (or autoregressive) outputs with extra loss.

    Args:
        loss_pos (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
    """
    loss_pos: Optional[torch.FloatTensor] = None
    output_pos: Optional[torch.FloatTensor] = None  # Output positional embeddings
    gt_pos: Optional[torch.FloatTensor] = None  # Ground truth positional embeddings
    output_pe_mask: Optional[torch.BoolTensor] = None  # Mask for output positional embeddings
    gt_coords_xy: Optional[torch.FloatTensor] = None  # Ground truth coordinates in xy plane
    last_hidden_state: Optional[torch.FloatTensor] = None  # Last hidden state of the model
    last_hidden_state_original: Optional[torch.FloatTensor] = None  # Original last hidden state before any modifications
    labels: Optional[torch.LongTensor] = None  # Labels for the loss calculation


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlavaModel(config)
        self.post_init()

        self.llm_hidden_dim = self.config.text_config.hidden_size
        self.l2_loss = nn.MSELoss()


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.FloatTensor] = None, # this is the PE for image tokens
        io_coords_pos: Optional[torch.Tensor] = None,
        loss_pos_lambda: Optional[torch.FloatTensor] = None,
        loss_for_pos: str ='l2',
        include_semantic_posemb = False,
        supervise_semantic_posemb = False,
        planning_only: Optional[bool] = False,
        single_coords_only: Optional[bool] = False,
        has_gt_planning: Optional[bool] = False,
        gt_coords_xy: Optional[torch.Tensor] = None,
        coords_encoder = None,
        coords_decoder = None, # NOTE: this is only for test inference
        ego_feature: Optional[torch.Tensor] = None, # shape (B, 1, hidden)
        enable_pe_input = False, # enable the use of PE in autoregressive manner, if False, only use PE for supervision
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CustomLlavaCausalLMOutputWithPast]:
        
        if pixel_values is not None:
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

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

            # if io_coords_pos.shape[0] != input_ids.shape[0]:
            #     raise ValueError(
            #         f"Positional embedding bs {io_coords_pos.shape[0]} does not match input_ids bs shape {input_ids.shape[0]}"
            #     )
            
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
            if io_coords_pos_indices.shape[0] == io_coords_pos.shape[1]:
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos
                io_coords_pos = scattered_io_coords_pos # now has shape (bs, len_input_ids, pos_dim)
            else:
                scattered_io_coords_pos[io_coords_pos_indices[:, 0], io_coords_pos_indices[:, 1]] = io_coords_pos[:, :io_coords_pos_indices.shape[0]]  # only use the first num_coords positions
                io_coords_pos = scattered_io_coords_pos

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
            pos_emb=pos_emb, # this is the PE for image tokens
            io_coords_pos_mask=io_coords_pos_mask, # This is a 0,1 mask
            enable_pe_input = enable_pe_input, # enable the use of PE in autoregressive manner, if False, only use PE for supervision
            io_coords_pos = io_coords_pos,
            include_semantic_posemb = include_semantic_posemb,
            ego_feature = ego_feature,
            **kwargs,
        )

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]


        hidden_states = outputs[0] # this is the last hidden states, shape (B, seq_len, hidden)
        hidden_states_original = hidden_states
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if hidden_states.dtype != self.lm_head.weight.dtype:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states[:, slice_indices, :])



        if io_coords_pos_mask is not None:
            # first create a mask to only select pos in answers, not in input.
            # which is position that are not IGNORE_INDEX
            if labels is None:
                output_mask = torch.ones_like(input_ids).to(hidden_states.device).bool().unsqueeze(-1).expand_as(hidden_states)
            else:
                output_mask = (labels != IGNORE_INDEX).unsqueeze(-1).expand_as(hidden_states)
            # io_coords_pos_mask come from selecting <EMB> in input_embeds
            # the corresponding selected output actually´represents <IND>, ('select this token'), (* PREDICT THIS TOKEN*)
            output_pe_mask = torch.roll(io_coords_pos_mask, shifts=-1, dims=-2) # roll one to left, output_pe_mask has shape (bs, len_input_ids, 2048) # this is the mask for the positions where we have io_coords_pos
            output_mask_hidden = torch.roll(output_mask, shifts=-1, dims=-2) # roll one to left,  has shape (bs, len_input_ids, 2048)
            output_pe_mask = output_pe_mask & output_mask_hidden # should be (bs, len_input_ids, hidden_states)
            output_pos = hidden_states.masked_select(output_pe_mask).view(-1, hidden_states.shape[-1])

            
            # select the ground truth pos from io_coords_pos
            gt_pos = io_coords_pos.masked_select(io_coords_pos_mask & output_mask).view(-1, io_coords_pos.shape[-1])

            if gt_coords_xy != None: 
                output_xy_mask = output_mask.masked_select(io_coords_pos_mask).view(1, -1, hidden_states.shape[-1])
                output_xy_mask = output_xy_mask[:,:,:2]
                
                gt_coords_xy = gt_coords_xy.masked_select(output_xy_mask).to(gt_pos.dtype)


        loss = None
        if labels is not None:

            # NOTE: 3. according to io_coords_pos_mask, replace it with -100
            if io_coords_pos_mask is not None:
                # Modify labels according to io_coords_pos_mask
                if not (supervise_semantic_posemb and include_semantic_posemb):
                    labels = labels.masked_fill(io_coords_pos_mask.squeeze(-1), IGNORE_INDEX)
                else:
                    print('llava forward: supervise_semantic_posemb is True, so do not modify labels') 

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

            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, num_items_in_batch=torch.tensor(B), **kwargs
            )

            if io_coords_pos_mask is not None:
                if loss_for_pos == 'l2_coords_mlp_2layer':
                    # use the mlp_1_coords to decode the output_pos and gt_pos
                    output_pos = output_pos.to(torch.float32) # make sure the output_pos is in the same dtype as mlp_1_coords
                    gt_coords_xy = gt_coords_xy.to(torch.float32)

                    loss_pos = torch.tensor(0.0, device=output_pos.device, dtype=output_pos.dtype)
                elif loss_for_pos == 'cosine':
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
                else:
                    raise ValueError(f"Unknown loss_for_pos: {loss_for_pos}. See documentation in config.py.")
            
                loss_pos = loss_pos * loss_pos_lambda



        if coords_encoder is not None and coords_decoder is not None:
            hs_dtype = hidden_states.dtype
            
            hidden_states = hidden_states.to(torch.float32) # make sure the hidden_states is in the same dtype as coords_encoder
            decoded_coords = coords_decoder(hidden_states)
            if decoded_coords.shape[-1] < 3:
                # append 0 in last dim 
                decoded_coords = torch.cat([decoded_coords, torch.zeros((*decoded_coords.shape[:-1],1), device=decoded_coords.device, dtype=decoded_coords.dtype)], dim=-1)
            decoded_coords[:,:,-1] = 0


            reencoded_pe = coords_encoder(decoded_coords)
            
            hidden_states = reencoded_pe.to(hs_dtype) # convert back to the original dtype

        if labels is not None and io_coords_pos_mask is not None:
            return CustomLlavaCausalLMOutputWithPast(
                loss=loss,
                loss_pos=loss_pos,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
                # extra args
                output_pos=output_pos,
                gt_pos=gt_pos,
                output_pe_mask=output_pe_mask,
                gt_coords_xy=gt_coords_xy,
                labels = labels
            )
        else:
            return CustomLlavaCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
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
            model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, inputs_embeds= inputs_embeds, **model_kwargs)
            
            if if_next_token_pos and not model_inputs['enable_pe_input']:
                # change the input_ids to POS_EMBEDDING_TOKEN_INDEX
                # NOTE: this is only for the next token position, not for the whole input_ids
                model_inputs["input_ids"] = torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=model_inputs["input_ids"].device).reshape(model_inputs["input_ids"].shape)
                # print('input_ids changed to POS_EMBEDDING_TOKEN_INDEX', self.tokenizer.batch_decode(input_ids, skip_special_tokens=False))
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
                # NOTE: remove the io_coords_pos from model_inputs to avoid error
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
                    # NOTE: update input_embeddings
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
                # NOTE (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
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
        
@dataclass
class CustomGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    hidden_states_for_output: Optional[torch.FloatTensor] = None  # Original last hidden state before any modifications




class CustomLlavaModel(LlavaModel):
    def __init__(self, config):
        super().__init__(config)
        self.image_pos_encoder = PositionalEncoding3D( # 2100 is the hidden dim of image backbone
            config.vision_config.hidden_size, 
            dtype_override=torch.float32, 
            freq_coeff=20000, # can be changed to 1000 or 100 or 10 for better performance
            pe_type='transformer', # 'transformer' or 'fone' or 'nerf'
            pe_scaling=0.2 # can be changed to 0.01 or 0.005 for better performance
        ) 
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: torch.Tensor = None,
        pos_emb: Optional[torch.FloatTensor] = None,
        io_coords_pos_mask: Optional[torch.Tensor] = None,
        io_coords_pos: Optional[torch.Tensor] = None,
        include_semantic_posemb: Optional[bool] = False,
        ego_feature: Optional[torch.Tensor] = None,
        enable_pe_input: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, LlavaModelOutputWithPast]:

        B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        if enable_pe_input and io_coords_pos is not None and io_coords_pos_mask is not None:
            if include_semantic_posemb:
                semantic_posemb = self.get_input_embeddings()(torch.tensor(POS_EMBEDDING_TOKEN_INDEX, device=io_coords_pos.device).unsqueeze(0)).squeeze(0).to(io_coords_pos.dtype) # semantic_posemb is the feature of the POS_EMBEDDING_TOKEN_INDEX token
                io_coords_pos = io_coords_pos + semantic_posemb # add the semantic posemb to the input pos emb
            inputs_embeds  = inputs_embeds.masked_scatter(io_coords_pos_mask, io_coords_pos)

        vision_feature_select_strategy = 'default'
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            # NOTE: HERE we add the pos_emb on top of the image embeddings
            if pos_emb is not None:
                pos_emb = pos_emb.reshape(image_features.shape)
                if pos_emb.shape[0] != image_features.shape[0]:
                    raise ValueError(
                        f"Positional embedding shape {pos_emb.shape} does not match image embedding shape {image_features.shape}"
                    )
                #print('add pos emb', pos_emb)
                image_features = image_features + pos_emb # shape [B, N, C]

            # NOTE: HERE starts the ego feature insertion
            if ego_feature is not None:
                image_features = image_features.reshape(B, -1, image_features.shape[-1]) # shape (B, N, C)
                image_features = torch.cat([image_features, ego_feature], dim=-2)
                image_features = image_features.reshape(-1, image_features.shape[-1]) # shape (B*N, C)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                n_image_tokens = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


