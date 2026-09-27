"""Optional batched Qwen execution around the UNMODIFIED CEE/VLM-PP policy.

Prefetch text for exactly the images the parent will process, then let the
parent perform all SAM, CEE selection, label replacement and confidence fusion.
No cross-image/iteration cache is introduced here. Batch size 1 is the fallback.
"""
from collections import deque
import logging
import torch
from torch.nn import functional as F
from model.semseg.vlm_pp import QwenVLPurifiedSemi


class BatchedQwenVLPurifiedSemi(QwenVLPurifiedSemi):
    def __init__(self, cfg, model, model_ema, class_names):
        self.qwen_batch_size = int(cfg.get('qwen_batch_size', 1))
        if self.qwen_batch_size < 1:
            raise ValueError('qwen_batch_size must be at least 1')
        self._prefetched = None
        super().__init__(cfg, model, model_ema, class_names)

    def _load_qwen_model(self):
        from transformers import BitsAndBytesConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        # DDP: every process owns one Qwen copy on its own GPU, rather than
        # every process's device_map='auto' spanning all visible GPUs.
        device_map = self.cfg.get('qwen_device_map', None)
        if device_map is None:
            distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
            device_map = {'': torch.cuda.current_device()} if distributed else 'auto'
        kwargs = dict(torch_dtype=torch.float16, device_map=device_map,
                      attn_implementation=self.cfg.get('qwen_attention', 'sdpa'))
        if self.use_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4')
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        self.qwen_model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
        self.processor.tokenizer.padding_side = 'left'

    def _run_qwen(self, messages):
        if self._prefetched is not None:
            if not self._prefetched:
                raise RuntimeError('CEE/VLM prefetch order mismatch')
            return self._prefetched.popleft()
        return super()._run_qwen(messages)

    def _run_batch(self, conversations):
        from qwen_vl_utils import process_vision_info
        texts = [self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True) for messages in conversations]
        images, videos = process_vision_info(conversations)
        inputs = self.processor(text=texts, images=images, videos=videos,
                                padding=True, return_tensors='pt').to(self.qwen_model.device)
        with torch.no_grad():
            output = self.qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        output = output[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(output, skip_special_tokens=True)

    def _generate_chunks(self, conversations):
        results = []
        for offset in range(0, len(conversations), self.qwen_batch_size):
            chunk = conversations[offset:offset + self.qwen_batch_size]
            outputs = None
            oom = False
            try:
                outputs = self._run_batch(chunk)
            except torch.cuda.OutOfMemoryError:
                if len(chunk) == 1:
                    raise
                oom = True
            # Outside the except block so the failed batch's traceback/tensors
            # are released before retrying with a smaller memory footprint.
            if oom:
                torch.cuda.empty_cache()
                logging.getLogger('global').warning('Qwen batch OOM; retrying this batch serially')
                outputs = [super(BatchedQwenVLPurifiedSemi, self)._run_qwen(x) for x in chunk]
            if len(outputs) != len(chunk):
                raise RuntimeError('Qwen output count does not match input count')
            results.extend(outputs)
        return results

    def get_qwen_purify(self, images, pseudo_labels, conf_scores, edge_map=None):
        if self.qwen_batch_size == 1:
            return super().get_qwen_purify(images, pseudo_labels, conf_scores, edge_map)

        # Read-only scheduling mirror. The parent's original implementation
        # remains the authority for all pixel masks and purification decisions.
        priority = conf_scores < self.vlm_pp_conf_threshold
        if self.use_cee and self.use_cee_vlm and edge_map is not None:
            edge = edge_map
            if tuple(edge.shape) != tuple(pseudo_labels.shape):
                edge = F.interpolate(edge.float().unsqueeze(1),
                                     size=pseudo_labels.shape[-2:], mode='nearest').squeeze(1).bool()
            priority = priority & edge
            if priority.sum().item() < self.min_edge_pixels:
                return super().get_qwen_purify(images, pseudo_labels, conf_scores, edge_map)
        infer = (not self.use_vlm_cache or
                 (self.current_iter + 1) % self.inference_interval == 0 or
                 self.cached_spatial_probs is None)
        indices = priority.flatten(1).any(1).nonzero().flatten().tolist() if infer else []
        if len(indices) < 2:
            return super().get_qwen_purify(images, pseudo_labels, conf_scores, edge_map)
        conversations = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': self._tensor_to_pil(images[b])},
            {'type': 'text', 'text': self.grounding_prompt}]}] for b in indices]
        self._prefetched = deque(self._generate_chunks(conversations))
        try:
            result = super().get_qwen_purify(images, pseudo_labels, conf_scores, edge_map)
            if self._prefetched:
                raise RuntimeError('Unused Qwen outputs: CEE/VLM scheduling mismatch')
            return result
        finally:
            self._prefetched = None
