"""
This script is adapted from the original implementation found at:
https://github.com/Juzezhang/language_of_motion

Author: Changan Chen, Juze Zhang and Shrinidhi Kowshika Lakshmikanth
Modified by: Wentao Jiang
License: Check the original repository for licensing details.
"""
from typing import List, Union, Dict, Any
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import Seq2SeqLMOutput
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        audio_codebook_size: int = 500,
        motion_framerate: float = 30.0,
        audio_samplerate: float = 16000.0,
        motion_down_sampling: int = 1,
        audio_down_sampling: int = 320,   ### audio down sample rate
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 512,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        flash_attention: bool = False,
        modalities: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.face_codebook_size = modalities['face']['codebook_size']
        self.hand_codebook_size = modalities['hand']['codebook_size']
        self.upper_codebook_size = modalities['upper']['codebook_size']
        self.lower_codebook_size = modalities['lower']['codebook_size']
        self.a_codebook_size = modalities['audio']['codebook_size']
        self.max_length = max_length
        self.motion_framerate = motion_framerate
        self.audio_samplerate = audio_samplerate
        self.motion_down_sampling = motion_down_sampling
        self.audio_down_sampling = audio_down_sampling
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.mask_ratio_audio = 0.08
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            if flash_attention:
                from turbot5 import T5ForConditionalGeneration
                self.language_model = T5ForConditionalGeneration.from_pretrained(
                    model_path, attention_type = 'flash',use_triton=True)
            else:
                from transformers import T5ForConditionalGeneration
                self.language_model = T5ForConditionalGeneration.from_pretrained(
                    model_path)
        else:
            raise ValueError("model_type must be either t5, llama, or mistral")

        for modality, settings in modalities.items():
            prefix = settings["prefix"]
            codebook_size = settings["codebook_size"] + 3
            # Generate tokens for the current modality
            tokens = [f"<{prefix}_{i}>" for i in range(codebook_size)]
            self.tokenizer.add_tokens(tokens)

        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # compute the fps ratio of audio tokens to motion tokens, used to build the causal cross attention mask
        self.audio_token_fps = audio_samplerate / audio_down_sampling  # 50
        self.motion_token_fps = motion_framerate / motion_down_sampling  # 30
        self.audio_to_motion_ratio = self.audio_token_fps / self.motion_token_fps  # ~1.67

        # cache the id range of audio tokens for quickly locating audio token positions in the encoder input
        self._audio_token_ids = None
        # cache the set of motion-related token ids
        self._motion_token_ids_cache = None

    def _get_audio_token_ids(self):
        """Get the set of ids for all audio tokens in the tokenizer"""
        if self._audio_token_ids is None:
            audio_token_ids = set()
            for i in range(self.a_codebook_size + 3):  # includes BOS, EOS and MASK tokens
                token_str = f"<audio_id_{i}>"
                token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                if token_id != self.tokenizer.unk_token_id:
                    audio_token_ids.add(token_id)
            self._audio_token_ids = audio_token_ids
        return self._audio_token_ids

    def _find_audio_token_positions(self, input_ids):
        """
        Find the positions of audio content tokens in the encoder input (excluding BOS and EOS markers).
        Use vectorized operations to avoid Python loops.

        Args:
            input_ids: [batch, seq_len] encoder input token ids

        Returns:
            audio_mask: [batch, seq_len] bool tensor, True means the position is an audio content token
            audio_cumsum: [batch, seq_len] cumulative count of audio content tokens before (and including) each position
            audio_content_counts: [batch] total number of audio content tokens in each batch
        """
        # audio BOS/EOS token id
        audio_bos_id = self.tokenizer.convert_tokens_to_ids(f"<audio_id_{self.a_codebook_size}>")
        audio_eos_id = self.tokenizer.convert_tokens_to_ids(f"<audio_id_{self.a_codebook_size + 1}>")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # find the positions of BOS and EOS, use cumsum to determine positions "inside the audio interval"
        is_bos = (input_ids == audio_bos_id)  # [batch, seq_len]
        is_eos = (input_ids == audio_eos_id)  # [batch, seq_len]

        # cumsum(bos) > cumsum(eos) means currently inside the audio interval
        bos_cumsum = is_bos.long().cumsum(dim=1)  # [batch, seq_len]
        eos_cumsum = is_eos.long().cumsum(dim=1)  # [batch, seq_len]
        in_audio = (bos_cumsum > eos_cumsum)  # [batch, seq_len]

        # audio content token = inside the audio interval and not BOS/EOS itself
        audio_mask = in_audio & (~is_bos) & (~is_eos)  # [batch, seq_len]

        # cumulative count of audio content tokens before (and including) each position
        audio_cumsum = audio_mask.long().cumsum(dim=1)  # [batch, seq_len]

        # total number of audio content tokens in each batch
        audio_content_counts = audio_mask.long().sum(dim=1)  # [batch]
        
        return audio_mask, audio_cumsum, audio_content_counts

    def _get_motion_token_ids_cache(self):
        """Get and cache the set of ids for all motion-related tokens"""
        if self._motion_token_ids_cache is None:
            cache = {}
            cache['motion_start_id'] = self.tokenizer.convert_tokens_to_ids("<motion_id_0>")
            cache['motion_end_id'] = self.tokenizer.convert_tokens_to_ids("<motion_id_1>")
            cache['face_bos_id'] = self.tokenizer.convert_tokens_to_ids(f"<face_id_{self.face_codebook_size}>")
            cache['face_eos_id'] = self.tokenizer.convert_tokens_to_ids(f"<face_id_{self.face_codebook_size+1}>")
            cache['hand_bos_id'] = self.tokenizer.convert_tokens_to_ids(f"<hand_id_{self.hand_codebook_size}>")
            cache['hand_eos_id'] = self.tokenizer.convert_tokens_to_ids(f"<hand_id_{self.hand_codebook_size+1}>")
            
            upper_token_ids = set()
            lower_token_ids = set()
            face_token_ids = set()
            hand_token_ids = set()
            for i in range(self.upper_codebook_size + 3):
                upper_token_ids.add(self.tokenizer.convert_tokens_to_ids(f"<upper_id_{i}>"))
            for i in range(self.lower_codebook_size + 3):
                lower_token_ids.add(self.tokenizer.convert_tokens_to_ids(f"<lower_id_{i}>"))
            for i in range(self.face_codebook_size + 3):
                face_token_ids.add(self.tokenizer.convert_tokens_to_ids(f"<face_id_{i}>"))
            for i in range(self.hand_codebook_size + 3):
                hand_token_ids.add(self.tokenizer.convert_tokens_to_ids(f"<hand_id_{i}>"))
            
            cache['upper_token_ids'] = upper_token_ids
            cache['lower_token_ids'] = lower_token_ids
            cache['face_token_ids'] = face_token_ids
            cache['hand_token_ids'] = hand_token_ids
            self._motion_token_ids_cache = cache
        return self._motion_token_ids_cache

    def _find_motion_token_timesteps(self, decoder_input_ids):
        """
        Find the motion timestep corresponding to each token in the decoder output.
        Use vectorized operations to avoid Python loops.

        In the compositional token format, the decoder output structure is:
        <motion_id_0> [<upper_id_x><lower_id_x>]* <motion_id_1> <face_id_BOS> [<face_id_x>]* <face_id_EOS> <hand_id_BOS> [<hand_id_x>]* <hand_id_EOS>

        Each timestep t corresponds to a pair of upper_id + lower_id tokens.
        The t-th token in the face and hand sequences also corresponds to timestep t.

        Args:
            decoder_input_ids: [batch, seq_len] decoder input/output token ids

        Returns:
            timestep_map: [batch, seq_len] the motion timestep (starting from 0) corresponding to each decoder position,
                          -1 for positions that are not motion tokens
        """
        cache = self._get_motion_token_ids_cache()
        motion_start_id = cache['motion_start_id']
        motion_end_id = cache['motion_end_id']
        face_bos_id = cache['face_bos_id']
        face_eos_id = cache['face_eos_id']
        hand_bos_id = cache['hand_bos_id']
        hand_eos_id = cache['hand_eos_id']

        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        ids = decoder_input_ids  # [batch, seq_len]

        # --- Motion part (upper+lower alternate, each lower pair completes one timestep) ---
        is_motion_start = (ids == motion_start_id)  # [batch, seq_len]
        is_motion_end = (ids == motion_end_id)
        # positions inside the motion interval (excluding start/end themselves)
        ms_cum = is_motion_start.long().cumsum(dim=1)
        me_cum = is_motion_end.long().cumsum(dim=1)
        in_motion = (ms_cum > me_cum) & (~is_motion_start)

        # lower token is the boundary of the timestep, use lower's cumsum as the timestep
        # build the mask for lower tokens (using the token id range)
        lower_min = self.tokenizer.convert_tokens_to_ids("<lower_id_0>")
        lower_max = self.tokenizer.convert_tokens_to_ids(f"<lower_id_{self.lower_codebook_size + 2}>")
        is_lower = (ids >= lower_min) & (ids <= lower_max) & in_motion

        # timestep within the motion interval = number of lower tokens before (and including) this position
        # but upper and lower alternate: upper_t, lower_t -> timestep t
        # timestep of upper_t = number of previous lower tokens = t
        # timestep of lower_t = number of previous lower tokens (excluding itself) = t
        # so timestep = cumsum(lower) - is_lower (i.e. lower itself is not counted in the current step)
        lower_cumsum = is_lower.long().cumsum(dim=1)  # [batch, seq_len]
        # need to recount within each motion interval: subtract the cumsum value at the motion_start position
        # find the starting cumsum of the motion interval each position belongs to
        # simplification: since there is usually only one motion interval, use the global cumsum directly
        motion_timestep = lower_cumsum - is_lower.long()  # both upper and lower correspond to the same timestep t

        # --- Face part ---
        is_face_bos = (ids == face_bos_id)
        is_face_eos = (ids == face_eos_id)
        fb_cum = is_face_bos.long().cumsum(dim=1)
        fe_cum = is_face_eos.long().cumsum(dim=1)
        in_face = (fb_cum > fe_cum) & (~is_face_bos)

        # timestep within the face interval = number of face content tokens before this position
        face_content = in_face & (~is_face_eos)
        face_cumsum = face_content.long().cumsum(dim=1)
        face_timestep = face_cumsum - face_content.long()  # timestep of the current token

        # --- Hand part ---
        is_hand_bos = (ids == hand_bos_id)
        is_hand_eos = (ids == hand_eos_id)
        hb_cum = is_hand_bos.long().cumsum(dim=1)
        he_cum = is_hand_eos.long().cumsum(dim=1)
        in_hand = (hb_cum > he_cum) & (~is_hand_bos)

        hand_content = in_hand & (~is_hand_eos)
        hand_cumsum = hand_content.long().cumsum(dim=1)
        hand_timestep = hand_cumsum - hand_content.long()

        # --- Merge ---
        # default to -1 (non-motion token)
        timestep_map = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

        # tokens within the motion interval (including start/end markers)
        motion_region = in_motion | is_motion_start | is_motion_end
        timestep_map[in_motion] = motion_timestep[in_motion]
        timestep_map[is_motion_start] = 0
        # motion_end corresponds to the last timestep
        timestep_map[is_motion_end] = motion_timestep[is_motion_end]

        # face interval
        face_region = in_face | is_face_bos | is_face_eos
        timestep_map[in_face] = face_timestep[in_face]
        timestep_map[is_face_bos] = 0
        timestep_map[is_face_eos] = face_timestep[is_face_eos]

        # hand interval
        hand_region = in_hand | is_hand_bos | is_hand_eos
        timestep_map[in_hand] = hand_timestep[in_hand]
        timestep_map[is_hand_bos] = 0
        timestep_map[is_hand_eos] = hand_timestep[is_hand_eos]

        return timestep_map

    def _build_causal_cross_attention_mask(self, source_input_ids, decoder_input_ids, source_attention_mask):
        """
        Build a causal cross attention mask so that when the decoder predicts the t-th motion timestep,
        it can only see audio tokens before the t-th timestep in the encoder.
        Use vectorized operations to avoid Python loops.

        Args:
            source_input_ids: [batch, enc_seq_len] encoder input token ids
            decoder_input_ids: [batch, dec_seq_len] decoder input token ids
            source_attention_mask: [batch, enc_seq_len] original encoder attention mask

        Returns:
            cross_attention_mask: [batch, dec_seq_len, enc_seq_len] causal cross attention mask
        """
        batch_size, enc_seq_len = source_input_ids.shape
        dec_seq_len = decoder_input_ids.shape[1]
        device = source_input_ids.device

        # find the positions of audio tokens in the encoder (vectorized)
        audio_mask, audio_cumsum, audio_content_counts = self._find_audio_token_positions(source_input_ids)
        # audio_mask: [batch, enc_seq_len] bool, True=audio content token
        # audio_cumsum: [batch, enc_seq_len] cumulative audio token count
        # audio_content_counts: [batch] total audio token count

        # find the motion timestep corresponding to each token in the decoder (vectorized)
        timestep_map = self._find_motion_token_timesteps(decoder_input_ids)
        # timestep_map: [batch, dec_seq_len], -1 means non-motion token

        # initialize the cross attention mask: first copy the original source_attention_mask
        # [batch, dec_seq_len, enc_seq_len]
        cross_attention_mask = source_attention_mask.unsqueeze(1).expand(-1, dec_seq_len, -1).clone().float()

        # compute the maximum number of audio tokens each decoder position can see
        # max_visible = ceil((t+1) * ratio), for t=-1 set a large value (can see all)
        t_clamped = timestep_map.clamp(min=0).float()  # [batch, dec_seq_len]
        max_visible_audio = torch.ceil((t_clamped + 1) * self.audio_to_motion_ratio).long()  # [batch, dec_seq_len]

        # for t=-1 positions (non-motion token), set a large value so it can see all audio tokens
        is_non_motion = (timestep_map == -1)  # [batch, dec_seq_len]
        max_visible_audio[is_non_motion] = enc_seq_len + 1  # large enough to not mask any audio

        # limit max_visible_audio to not exceed the total audio token count
        # audio_content_counts: [batch] -> [batch, 1]
        max_visible_audio = torch.min(max_visible_audio, audio_content_counts.unsqueeze(1).expand_as(max_visible_audio))

        # for each audio position in the encoder, determine whether it should be visible to the current decoder position
        # audio_cumsum[b, enc_pos] is the number of audio tokens before (and including) encoder position enc_pos
        # if audio_cumsum[b, enc_pos] > max_visible_audio[b, dec_pos], then that audio token is not visible
        #
        # audio_cumsum: [batch, enc_seq_len] -> [batch, 1, enc_seq_len]
        # max_visible_audio: [batch, dec_seq_len] -> [batch, dec_seq_len, 1]
        audio_cumsum_expanded = audio_cumsum.unsqueeze(1)  # [batch, 1, enc_seq_len]
        max_visible_expanded = max_visible_audio.unsqueeze(2)  # [batch, dec_seq_len, 1]

        # for audio positions, mask out if its cumsum > max_visible
        # note: only apply this mask to encoder positions where audio_mask is True
        audio_mask_expanded = audio_mask.unsqueeze(1).float()  # [batch, 1, enc_seq_len]
        should_mask = (audio_cumsum_expanded > max_visible_expanded).float()  # [batch, dec_seq_len, enc_seq_len]

        # only mask audio positions
        mask_to_apply = should_mask * audio_mask_expanded  # [batch, dec_seq_len, enc_seq_len]

        # apply the mask
        cross_attention_mask = cross_attention_mask * (1.0 - mask_to_apply)

        return cross_attention_mask

    def forward(self, 
                texts: List[str], 
                # text_timestamp: Optional[List[str]] = None,
                body_tokens: Optional[Dict[str, Tensor]] = None,
                audio_data: Optional[Dict[str, Tensor]] = None,
                lengths: Optional[Dict[str, List[int]]] = None,
                context: Optional[Dict[str, Any]] = None,
                tasks: Optional[dict] = None, 
                emotion_label: Optional[List[str]] = None):

        # Extract body tokens
        face_tokens = body_tokens.get('face') if body_tokens else None
        hand_tokens = body_tokens.get('hand') if body_tokens else None
        upper_tokens = body_tokens.get('upper') if body_tokens else None
        lower_tokens = body_tokens.get('lower') if body_tokens else None
        
        # Extract audio data
        audio_tokens = audio_data.get('tokens')
        # Extract length information
        motion_lengths = lengths.get('motion')
        audio_lengths = lengths.get('audio')
        # Extract context information
        emotion_label = context.get('emotion_label')

        face_strings, hand_strings, upper_strings, lower_strings, motion_string = self.compositional_motion_token_to_string(face_tokens, hand_tokens, upper_tokens, lower_tokens, motion_lengths)
        audio_strings = self.audio_token_to_string(audio_tokens, audio_lengths)
        inputs, outputs = self.template_fulfill(tasks, motion_lengths, audio_lengths, face_strings, hand_strings, upper_strings, lower_strings, motion_string, audio_strings, texts, emotion_label)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(face_tokens.device)
        source_input_ids = source_encoding.input_ids.to(face_tokens.device)

        target_inputs = self.tokenizer(outputs,
                                        padding='max_length',
                                        max_length=self.max_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors="pt")

        labels_input_ids = target_inputs.input_ids.to(face_tokens.device)
        lables_attention_mask = target_inputs.attention_mask.to(
            face_tokens.device)

        labels_input_ids[labels_input_ids == 0] = -100

        # build the decoder input (T5 uses labels shifted right by one as decoder_input_ids)
        decoder_input_ids = self.language_model._shift_right(labels_input_ids.clamp(min=0))

        # build the causal cross attention mask
        # [batch, dec_seq_len, enc_seq_len]
        causal_cross_mask = self._build_causal_cross_attention_mask(
            source_input_ids, decoder_input_ids, source_attention_mask
        )

        # manually call the encoder
        encoder_outputs = self.language_model.encoder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
        )

        # pass the 3D causal cross attention mask [batch, dec_seq_len, enc_seq_len]
        # T5's decoder stack will automatically call invert_attention_mask to convert it to
        # [batch, 1, dec_seq_len, enc_seq_len] and perform inversion (1->0, 0->-inf)
        # in causal_cross_mask: 1 means visible, 0 means invisible

        # manually call the decoder
        decoder_outputs = self.language_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=lables_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=causal_cross_mask,
        )

        # compute lm_head output and loss
        sequence_output = decoder_outputs[0]
        if self.language_model.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.language_model.model_dim**-0.5)
        lm_logits = self.language_model.lm_head(sequence_output)

        loss = None
        if labels_input_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels_input_ids.view(-1))

        outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        return outputs
    
    def generate_direct(self,
                        input: List[str] = None,
                        max_length: int = 512,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device

        # Tokenize
        source_encoding = self.tokenizer(input,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        # use causal autoregressive generation
        outputs = self._generate_causal_autoregressive(
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            max_length=max_length,
            do_sample=do_sample,
        )
        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)

        face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text = self.motion_string_to_compositional_token(outputs_string)
        return face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text

    @torch.no_grad()
    def _generate_causal_autoregressive(self, source_input_ids, source_attention_mask,
                                         max_length=512, do_sample=True, temperature=1.0):
        """
        Causal autoregressive generation: only after predicting one compositional token is the corresponding
        timestep's audio token exposed.
        Use vectorized operations to avoid Python loops.

        Args:
            source_input_ids: [batch, enc_seq_len] encoder input
            source_attention_mask: [batch, enc_seq_len] encoder attention mask
            max_length: maximum generation length
            do_sample: whether to sample
            temperature: sampling temperature

        Returns:
            generated_ids: [batch, gen_seq_len] generated token sequence
        """
        batch_size = source_input_ids.shape[0]
        enc_seq_len = source_input_ids.shape[1]
        device = source_input_ids.device

        # find the positions of audio tokens in the encoder (vectorized)
        audio_mask, audio_cumsum, audio_content_counts = self._find_audio_token_positions(source_input_ids)
        # audio_mask: [batch, enc_seq_len] bool
        # audio_cumsum: [batch, enc_seq_len]
        # audio_content_counts: [batch]

        # first run the encoder with the full encoder attention mask (encoder self-attention does not need a causal mask)
        encoder_outputs = self.language_model.encoder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # initialize the decoder input (T5 starts with decoder_start_token_id)
        decoder_start_token_id = self.language_model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.language_model.config.pad_token_id

        generated_ids = torch.full(
            (batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device
        )

        # EOS token id
        eos_token_id = self.language_model.config.eos_token_id

        # track whether each sample is finished
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        # precompute constants for mask construction
        audio_mask_float = audio_mask.float()  # [batch, enc_seq_len]
        audio_mask_expanded_enc = audio_mask_float.unsqueeze(1)  # [batch, 1, enc_seq_len]
        audio_cumsum_expanded = audio_cumsum.unsqueeze(1)  # [batch, 1, enc_seq_len]

        # autoregressive generation
        for step in range(max_length - 1):
            if not unfinished.any():
                break

            current_dec_len = generated_ids.shape[1]

            # compute the motion timestep corresponding to each decoder position (vectorized)
            timestep_map = self._find_motion_token_timesteps(generated_ids)
            # timestep_map: [batch, current_dec_len]

            # build the 3D cross attention mask (vectorized)
            # initialize as an expansion of source_attention_mask
            cross_mask_3d = source_attention_mask.unsqueeze(1).expand(
                -1, current_dec_len, -1
            ).clone().float()

            # compute the maximum number of audio tokens each decoder position can see
            t_clamped = timestep_map.clamp(min=0).float()  # [batch, current_dec_len]
            max_visible = torch.ceil((t_clamped + 1) * self.audio_to_motion_ratio).long()

            # non-motion tokens can see all audio
            is_non_motion = (timestep_map == -1)
            max_visible[is_non_motion] = enc_seq_len + 1

            # limit to not exceed the total audio token count
            max_visible = torch.min(max_visible, audio_content_counts.unsqueeze(1).expand_as(max_visible))

            # build the mask: positions where audio_cumsum > max_visible need to be masked out
            max_visible_expanded = max_visible.unsqueeze(2)  # [batch, current_dec_len, 1]
            should_mask = (audio_cumsum_expanded.expand(-1, current_dec_len, -1) > max_visible_expanded).float()
            mask_to_apply = should_mask * audio_mask_expanded_enc.expand(-1, current_dec_len, -1)
            cross_mask_3d = cross_mask_3d * (1.0 - mask_to_apply)

            # pass the 3D mask [batch, dec_seq_len, enc_seq_len]
            # T5's decoder stack will automatically call invert_attention_mask
            decoder_outputs = self.language_model.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_mask_3d,
                use_cache=False,
            )

            # get the logits
            sequence_output = decoder_outputs[0]
            if self.language_model.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.language_model.model_dim**-0.5)
            lm_logits = self.language_model.lm_head(sequence_output)

            # take the logits at the last position
            next_token_logits = lm_logits[:, -1, :]

            # sample or greedy
            if do_sample:
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # for finished samples, replace with pad token
            next_tokens = next_tokens * unfinished + (1 - unfinished.long()) * self.language_model.config.pad_token_id

            # update the generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # check whether EOS was generated
            unfinished = unfinished & (next_tokens != eos_token_id)

        return generated_ids

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             body_tokens: Optional[Dict[str, Tensor]] = None,
                             audio_data: Optional[Dict[str, Tensor]] = None,
                             lengths: Optional[Dict[str, List[int]]] = None,
                             context: Optional[Dict[str, Any]] = None,
                             task: str = "t2m",
                             stage: str = 'train',
                             tasks: dict = None):
        """
        Generate motion tokens conditioned on various inputs with simplified parameter structure.
        
        Parameters:
        -----------
        texts: Optional[List[str]]
            Text prompts for text-to-motion generation
        
        body_tokens: Optional[Dict[str, Tensor]]
            Dictionary containing tokens for different body parts:
            {
                'face': face_tokens,
                'hand': hand_tokens,
                'upper': upper_tokens,
                'lower': lower_tokens
            }
        
        audio_data: Optional[Dict[str, Tensor]]
            Dictionary containing audio-related data:
            {
                'tokens': audio tokens,
                'onset': onset data,
                'amplitude_envelope': amplitude data,
                'timestamps': text timestamp alignment data
            }
            
        lengths: Optional[Dict[str, List[int]]]
            Dictionary containing length information:
            {
                'motion': motion sequence lengths,
                'audio': audio sequence lengths
            }
            
        context: Optional[Dict[str, Any]]
            Dictionary containing additional contextual information:
            {
                'combine_strings': combined textual representation,
                'emotion_label': emotion labels,
                'text_timestamp': timestamps for text
            }
            
        task: str
            Task type ('t2m', 'm2m', 'pred', 'inbetween', 'a2m', 'at2m', 'm2e', 'm2t', 'a2t')
            
        with_len: bool
            Whether to include length in the prompt
            
        stage: str
            Training stage
            
        tasks: dict
            Task specifications
        """
        self.device = self.language_model.device
        
        # Set default values for required dictionaries
        if context is None:
            context = {}
        if lengths is None:
            lengths = {}
        if audio_data is None:
            audio_data = {}
        
        # Extract body tokens
        face_tokens = body_tokens.get('face') if body_tokens else None
        hand_tokens = body_tokens.get('hand') if body_tokens else None
        upper_tokens = body_tokens.get('upper') if body_tokens else None
        lower_tokens = body_tokens.get('lower') if body_tokens else None
        
        # Extract audio data
        audio_tokens = audio_data.get('tokens')

        # Extract length information
        motion_lengths = lengths.get('motion')
        audio_lengths = lengths.get('audio')
        
        # Extract context information
        emotion_label = context.get('emotion_label')
        
        # Rest of the function implementation
        if task in ["t2m", "a2m"]:
            # Initialize string variables
            batch_size = 0
            # Determine batch size from available inputs
            if texts is not None:
                batch_size = len(texts)
            elif audio_tokens is not None:
                batch_size = len(audio_tokens)
            elif face_tokens is not None:
                batch_size = len(face_tokens)
            
            # Initialize empty strings for all inputs
            motion_strings = [''] * batch_size
            audio_strings = [''] * batch_size
            face_strings = [''] * batch_size
            hand_strings = [''] * batch_size
            upper_strings = [''] * batch_size
            lower_strings = [''] * batch_size
            combine_strings = [''] * batch_size
            emotion_strings = [''] * batch_size if emotion_label is None else emotion_label
            
            # Task-specific processing
            if task == "t2m":
                assert texts is not None, "Text input required for t2m task"
                audio_lengths = [0] * batch_size
                tasks = [{
                    'input': ['Generate motion: <Caption_Placeholder>'],
                    'output': ['']
                }] * batch_size
                lengths = [0] * batch_size
                
            elif task == "a2m":
                assert audio_tokens is not None, "Audio tokens required for a2m task"
                audio_strings = self.audio_token_to_string(audio_tokens, audio_lengths)
                # tasks = [{
                #     'input': ["Generate face motion: <AudioTranscript_Placeholder>"],
                #     'output': ['']
                # }] * batch_size
                tasks = [{
                    'input': ["Based on <Audio_Placeholder>, generate a synchronized movement sequence involving both upper, lower, face and hands body."],
                    'output': ['']
                }] * batch_size
                lengths = [0] * batch_size
            elif task == "at2m":
                assert audio_tokens is not None, "Audio tokens required for at2m task"
                audio_lengths = [0] * batch_size if audio_lengths is None else audio_lengths
                combine_strings = self.audio_transcript_token_to_string(audio_tokens, text_timestamp, audio_lengths)
                tasks = [{
                    'input': [
                        "Given the audio and transcript with precise timestamp alignment in \"<AudioTranscript_Placeholder>\", generate a coordinated motion sequence involving face, hand, upper, and lower body movements."
                    ],
                    'output': ['']
                }] * batch_size
                
                lengths = [0] * batch_size

            # Create inputs and outputs from templates
            inputs, outputs = self.template_fulfill(
                tasks, lengths, audio_lengths,
                face_strings, hand_strings, upper_strings, lower_strings, 
                motion_strings, audio_strings, texts,
                combine_strings, emotion_strings
            )
            
            # Generate tokens using the language model
            face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text = self.generate_direct(
                inputs, max_length=self.max_length, num_beams=1, do_sample=True
            )
            
            # Return generated tokens as a dictionary for consistency
            return {
                'face': face_tokens,
                'hand': hand_tokens, 
                'upper': upper_tokens, 
                'lower': lower_tokens, 
                'text': cleaned_text
            }
        
    def compositional_motion_token_to_string(self, face_token: Tensor, hand_token: Tensor, upper_token: Tensor, lower_token: Tensor, lengths: List[int]):
        motion_string = []
        face_string = []
        hand_string = []
        upper_string = []
        lower_string = []

        # motion_string.append('<motion_id_0>')
        for i in range(len(lengths)):
            face_i = face_token[i].cpu() if face_token[i].device.type == 'cuda' else face_token[i]
            hand_i = hand_token[i].cpu() if hand_token[i].device.type == 'cuda' else hand_token[i]
            upper_i = upper_token[i].cpu() if upper_token[i].device.type == 'cuda' else upper_token[i]
            lower_i = lower_token[i].cpu() if lower_token[i].device.type == 'cuda' else lower_token[i]
            face_list = face_i.tolist()[:lengths[i]]
            hand_list = hand_i.tolist()[:lengths[i]]
            upper_list = upper_i.tolist()[:lengths[i]]
            lower_list = lower_i.tolist()[:lengths[i]]

            face_string_tmp = f'<face_id_{self.face_codebook_size}>'
            for j in range(lengths[i]):
                face_string_tmp = face_string_tmp + ''.join(f'<face_id_{int(face_list[j])}>')
            face_string_tmp += f'<face_id_{self.face_codebook_size+1}>'
            face_string.append(face_string_tmp)

            hand_string_tmp = f'<hand_id_{self.hand_codebook_size}>'
            for j in range(lengths[i]):
                hand_string_tmp = hand_string_tmp + ''.join(f'<hand_id_{int(hand_list[j])}>')
            hand_string_tmp += f'<hand_id_{self.hand_codebook_size+1}>'
            hand_string.append(hand_string_tmp)

            upper_string_tmp = f'<upper_id_{self.upper_codebook_size}>'
            for j in range(lengths[i]):
                upper_string_tmp = upper_string_tmp + ''.join(f'<upper_id_{int(upper_list[j])}>')
            upper_string_tmp += f'<upper_id_{self.upper_codebook_size+1}>'
            upper_string.append(upper_string_tmp)

            lower_string_tmp = f'<lower_id_{self.lower_codebook_size}>'
            for j in range(lengths[i]):
                lower_string_tmp = lower_string_tmp + ''.join(f'<lower_id_{int(lower_list[j])}>')
            lower_string_tmp += f'<lower_id_{self.lower_codebook_size+1}>'
            lower_string.append(lower_string_tmp)

            motion_string_tmp = '<motion_id_0>'
            for j in range(lengths[i]):
                motion_string_tmp = motion_string_tmp  + ''.join(f'<upper_id_{int(upper_list[j])}>') + ''.join(f'<lower_id_{int(lower_list[j])}>')
            motion_string_tmp += '<motion_id_1>'
            motion_string.append(motion_string_tmp)

        return face_string, hand_string, upper_string, lower_string, motion_string

    def audio_token_to_string(self, audio_token: Tensor, lengths: List[int]):
        audio_string = []
        for i in range(len(audio_token)):
            if audio_token[i] is None:
                continue
            audio_i = audio_token[i].cpu() if audio_token[i].device.type == 'cuda' else audio_token[i]
            audio_list = audio_i.tolist()[:lengths[i]]

            audio_string_tmp = f'<audio_id_{self.a_codebook_size}>'
            for j in range(lengths[i]):
                audio_string_tmp += ''.join(f'<audio_id_{int(audio_list[j])}>')
            audio_string_tmp += f'<audio_id_{self.a_codebook_size + 1}>'

            audio_string.append(audio_string_tmp)

        return audio_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_compositional_token(self, motion_string: List[str]):
        face_tokens = []
        hand_tokens = []
        lower_tokens = []
        upper_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str_emage(motion_string[i], '<motion_id_0>','<motion_id_1>')
            if string == '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>':

                face_string = self.get_middle_str_emage_v2(motion_string[i], f'<face_id_{self.face_codebook_size}>',f'<face_id_{self.face_codebook_size+1}>')
                hand_string = self.get_middle_str_emage_v2(motion_string[i], f'<hand_id_{self.hand_codebook_size}>',f'<hand_id_{self.hand_codebook_size+1}>')
                upper_string = self.get_middle_str_emage_v2(motion_string[i], f'<upper_id_{self.upper_codebook_size}>',f'<upper_id_{self.upper_codebook_size+1}>')
                lower_string = self.get_middle_str_emage_v2(motion_string[i], f'<lower_id_{self.lower_codebook_size}>',f'<lower_id_{self.lower_codebook_size+1}>')

                # string_list = string.split('><')
                face_string_list = face_string.split('><')
                hand_string_list = hand_string.split('><')
                upper_string_list = upper_string.split('><')
                lower_string_list = lower_string.split('><')

                face_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in face_string_list[1:-1] if i.startswith('face') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                hand_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in hand_string_list[1:-1] if i.startswith('hand') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                upper_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in upper_string_list[1:-1] if i.startswith('upper') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                lower_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in lower_string_list[1:-1] if i.startswith('lower') and i.split('_')[-1].replace('>', '').isdigit()
                ]

            else:
                string_list = string.split('><')
                face_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('face') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                hand_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('hand') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                lower_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('lower') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                upper_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('upper') and i.split('_')[-1].replace('>', '').isdigit()
                ]


            if len(face_token_list) == 0:
                face_token_list = [0]
            if len(hand_token_list) == 0:
                hand_token_list = [0]
            if len(lower_token_list) == 0:
                lower_token_list = [0]
            if len(upper_token_list) == 0:
                upper_token_list = [0]

            face_token_list = torch.tensor(face_token_list, dtype=int).to(self.device)
            hand_token_list = torch.tensor(hand_token_list, dtype=int).to(self.device)
            lower_token_list = torch.tensor(lower_token_list, dtype=int).to(self.device)
            upper_token_list = torch.tensor(upper_token_list, dtype=int).to(self.device)

            face_tokens.append(face_token_list)
            hand_tokens.append(hand_token_list)
            lower_tokens.append(lower_token_list)
            upper_tokens.append(upper_token_list)

            if string == '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>':
                output_string.append(motion_string[i].replace(face_string, '<Face_Placeholder>')
                                     .replace(hand_string, '<Hand_Placeholder>')
                                     .replace(upper_string, '<Upper_Placeholder>')
                                     .replace(lower_string, '<Lower_Placeholder>'))
            else:
                output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))

        return face_tokens, hand_tokens, upper_tokens, lower_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, audio_length: int,
                                face_string: str, hand_string: str, upper_string: str,lower_string: str, motion_string: str,
                                audio_string: str, text: str, emotion_label: str):

        seconds = math.floor(length / self.motion_framerate)
        motion_splited = motion_string.split('>')
        face_splited = face_string.split('>')
        hand_splited = hand_string.split('>')
        upper_splited = upper_string.split('>')
        lower_splited = lower_string.split('>')
        audio_splited = audio_string.split('>')

        motion_token_length = length / self.motion_down_sampling

        # audio_token_length = audio_length / self.audio_down_sampling
        predict_head = int(motion_token_length * self.predict_ratio + 1)


        # Randomly choose the starting position and the length of the mask region
        mask_length = int(motion_token_length * self.inbetween_ratio)  # Calculate the length of the masked region
        start_index = random.randint(0,  int(motion_token_length - mask_length))  # Randomly select the starting index for masking
        # Ensure the mask region is within the bounds of the sequence
        masked_head = start_index  # The starting index of the masked region
        masked_tail = start_index + mask_length  # The ending index of the masked region

        mask_length_audio = int(audio_length * self.mask_ratio_audio)  # Calculate the length of the masked region
        start_index_audio = random.randint(0,  audio_length - mask_length_audio)  # Randomly select the starting index for masking
        masked_head_audio = start_index_audio  # The starting index of the masked region
        masked_tail_audio = start_index_audio + mask_length_audio  # The ending index of the masked region



        motion_predict_head = '>'.join(motion_splited[:predict_head]) + f'><motion_id_1>'
        motion_predict_last = f'<motion_id_0>' + '>'.join(motion_splited[predict_head:])
        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_2>' * (masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        face_predict_head = '>'.join(face_splited[:predict_head]) + f'><face_id_{self.face_codebook_size+1}>'
        face_predict_last = f'<face_id_{self.face_codebook_size}>' + '>'.join(face_splited[predict_head:])
        face_masked = '>'.join(
            face_splited[:masked_head]
        ) + '>' + f'<face_id_{self.face_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(face_splited[masked_tail:])

        hand_predict_head = '>'.join(hand_splited[:predict_head]) + f'><hand_id_{self.hand_codebook_size+1}>'
        hand_predict_last = f'<hand_id_{self.hand_codebook_size}>' + '>'.join(hand_splited[predict_head:])
        hand_masked = '>'.join(
            hand_splited[:masked_head]
        ) + '>' + f'<hand_id_{self.hand_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(hand_splited[masked_tail:])

        upper_predict_head = '>'.join(upper_splited[:predict_head]) + f'><upper_id_{self.upper_codebook_size+1}>'
        upper_predict_last = f'<upper_id_{self.upper_codebook_size}>' + '>'.join(upper_splited[predict_head:])
        upper_masked = ('>'.join(upper_splited[:masked_head]) + '>'
                        + f'<upper_id_{self.upper_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(upper_splited[masked_tail:]))

        lower_predict_head = '>'.join(lower_splited[:predict_head]) + f'><lower_id_{self.lower_codebook_size+1}>'
        lower_predict_last = f'<lower_id_{self.lower_codebook_size}>' + '>'.join(lower_splited[predict_head:])
        lower_masked = '>'.join(
            lower_splited[:masked_head]
        ) + '>' + f'<lower_id_{self.lower_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(lower_splited[masked_tail:])


        audio_masked = '>'.join(
            audio_splited[:masked_head_audio]
        ) + '>' + f'<audio_id_{self.a_codebook_size+2}>' * (masked_tail_audio - masked_head_audio) + '>'.join(audio_splited[masked_tail_audio:])


        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        if text == None:
            text = f'\"{text}\"'
        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Transcript_Placeholder>', text).replace(
            '<Emotion_Placeholder>', emotion_label).replace(
            '<Face_Placeholder>', face_string).replace(
            '<Hand_Placeholder>', hand_string).replace(
            '<Upper_Placeholder>', upper_string).replace(
            '<Lower_Placeholder>', lower_string).replace(
            '<Motion_Placeholder>', motion_string).replace(
            '<Audio_Placeholder>', audio_string).replace(
            '<Frame_Placeholder>',f'{length}').replace(
            '<Second_Placeholder>', '%.1f' % seconds).replace(
            '<Face_Placeholder_s1>', face_predict_head).replace(
            '<Hand_Placeholder_s1>', hand_predict_head).replace(
            '<Upper_Placeholder_s1>', upper_predict_head).replace(
            '<Lower_Placeholder_s1>', lower_predict_head).replace(
            '<Face_Placeholder_s2>', face_predict_last).replace(
            '<Hand_Placeholder_s2>', hand_predict_last).replace(
            '<Upper_Placeholder_s2>', upper_predict_last).replace(
            '<Lower_Placeholder_s2>', lower_predict_last).replace(
            '<Face_Placeholder_Masked>', face_masked).replace(
            '<Hand_Placeholder_Masked>', hand_masked).replace(
            '<Upper_Placeholder_Masked>', upper_masked).replace(
            '<Lower_Placeholder_Masked>', lower_masked).replace(
            '<Audio_Placeholder_Masked>', audio_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         audio_lengths,
                         face_strings,
                         hand_strings,
                         upper_strings,
                         lower_strings,
                         motion_string,
                         audio_strings,
                         texts,
                         emotion_label,
                         stage='test'):
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i], lower_strings[i], motion_string[i],
                                             audio_strings[i], texts[i], emotion_label[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i], lower_strings[i], motion_string[i],
                                             audio_strings[i], texts[i], emotion_label[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def get_middle_str_emage(self, content, startStr, endStr):

        try:
            startIndex = content.index(startStr)
        except:
            return '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>'

        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return '<motion_id_0>' + content[startIndex:] + '<motion_id_1>'

        return '<motion_id_0>' + content[startIndex:endIndex] + '<motion_id_1>'

    def get_middle_str_emage_v2(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
        except:
            return startStr + '<face_id_0><hand_id_0><upper_id_0><lower_id_0>' + endStr
        
        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return startStr + content[startIndex:] + endStr

        return startStr + content[startIndex:endIndex] + endStr
