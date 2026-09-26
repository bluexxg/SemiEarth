import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from util.classes import CLASSES
import json
import re
from PIL import Image
import numpy as np


def _is_rank0():
    """Rank-0 check so debug logs are not duplicated across DDP workers."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

class QwenVLPurifiedSemi:
    def __init__(self, cfg, model, model_ema, class_names):
        self.model = model
        self.model_ema = model_ema
        self.cfg = cfg
        self.class_names = class_names
        self.num_classes = len(class_names)

        self.vlm_pp_conf_threshold = cfg.get('vlm_pp_conf_threshold', 0.7)
        self.model_name = cfg.get('qwen_model_name', 'Qwen/Qwen2.5-VL-3B-Instruct')
        self.use_4bit = cfg.get('use_4bit', True)
        self.bbox_confidence = cfg.get('bbox_confidence', 0.95)

        self.inference_interval = cfg.get('qwen_inference_interval', 2)
        self.current_iter = 0

        self.cached_spatial_probs = None
        self.cached_shape = None

        # ---- CEE-guided edge-aware purification --------------------------
        self.use_cee = cfg.get('use_cee', False)
        self.use_cee_vlm = cfg.get('use_cee_vlm', True)
        self.edge_vlm_weight = cfg.get('edge_vlm_weight', 1.5)
        self.cee_debug = cfg.get('cee_debug', False)
        self.min_edge_pixels = cfg.get('min_edge_pixels', 0)
        # Caching the VLM spatial probs across iterations is only safe when the
        # set of "images to refine" does not change between batches. With CEE
        # the priority mask (= low-conf & edge) changes every iteration, so by
        # default we turn the cache off under CEE-guided mode and keep the
        # original interval-caching behaviour for the baseline (use_cee=False).
        _vlm_cache = cfg.get('use_vlm_cache', None)
        if _vlm_cache is None:
            self.use_vlm_cache = not (self.use_cee and self.use_cee_vlm)
        else:
            self.use_vlm_cache = bool(_vlm_cache)

        self.use_vlm_on_mismatch = cfg.get('use_vlm_on_mismatch', True)

        self.use_sam = cfg.get('use_sam', True)
        self.sam_model_type = cfg.get('sam_model_type', 'vit_b')  # vit_b, vit_l, vit_h
        self.sam_checkpoint = cfg.get('sam_checkpoint', 'pretrained/sam_vit_b_01ec64.pth')

        self._load_qwen_model()

        if self.use_sam:
            self._load_sam_model()

        self.grounding_prompt = f"""Locate all objects in this image.
    Classes: {', '.join(self.class_names)}
    For each detected object, output: class_name: [x1, y1, x2, y2]"""

        print(f"初始化完成")

    def _load_qwen_model(self):
        from transformers import BitsAndBytesConfig
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.qwen_model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )

    def _load_sam_model(self):
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.qwen_model.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        print(f"  - SAM模型: {self.sam_model_type}")

    def _tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = tensor.clamp(0, 1)

        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        numpy_image = numpy_image.transpose(1, 2, 0)
        return Image.fromarray(numpy_image)

    def _run_qwen(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_model.device)

        with torch.no_grad():
            generated_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )[0]

    def _parse_output(self, output_text):
        valid_detections = {}  # {class_name: [(x1,y1,x2,y2), ...]}
        output_lower = output_text.lower()

        for class_name in self.class_names:
            if class_name.lower() in output_lower:
                pattern = rf"{re.escape(class_name)}[:\s]*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
                matches = re.findall(pattern, output_text, re.IGNORECASE)
                if matches:
                    valid_detections[class_name] = [tuple(map(int, m)) for m in matches]

        return valid_detections

    def _grounding_inference(self, pil_image, feat_H, feat_W):
        img_W, img_H = pil_image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.grounding_prompt},
                ],
            }
        ]

        output_text = self._run_qwen(messages)

        valid_detections = self._parse_output(output_text)

        if not valid_detections:
            return torch.zeros(self.num_classes, feat_H, feat_W)

        spatial_probs = torch.zeros(self.num_classes, feat_H, feat_W)

        scale_x = feat_W / img_W
        scale_y = feat_H / img_H

        if self.use_sam:
            np_image = np.array(pil_image)
            self.sam_predictor.set_image(np_image)

        for class_name, bboxes in valid_detections.items():
            idx = self.class_names.index(class_name)

            for (x1, y1, x2, y2) in bboxes:
                if self.use_sam:
                    box = np.array([x1, y1, x2, y2])
                    masks, scores, _ = self.sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    best_mask = masks[scores.argmax()]
                    mask_tensor = torch.from_numpy(best_mask).float().unsqueeze(0).unsqueeze(0)
                    mask_resized = F.interpolate(mask_tensor, size=(feat_H, feat_W), mode='nearest')[0, 0]
                    spatial_probs[idx] = torch.maximum(
                        spatial_probs[idx],
                        mask_resized * self.bbox_confidence
                    )
                else:
                    fx1 = int(x1 * scale_x)
                    fx2 = int(x2 * scale_x)
                    fy1 = int(y1 * scale_y)
                    fy2 = int(y2 * scale_y)
                    fx1, fx2 = max(0, fx1), min(feat_W, fx2)
                    fy1, fy2 = max(0, fy1), min(feat_H, fy2)

                    if fx2 > fx1 and fy2 > fy1:
                        spatial_probs[idx, fy1:fy2, fx1:fx2] = self.bbox_confidence
        spatial_probs = F.softmax(spatial_probs, dim=0)
        return spatial_probs

    def get_qwen_purify(self, images, pseudo_labels, conf_scores, edge_map=None):
        """CEE-guided VLM purification.

        Backward compatible: when ``edge_map is None`` (or ``use_cee=False``)
        this behaves exactly like the original VLM-PP purification.

        When ``use_cee=True & use_cee_vlm=True & edge_map is not None``:
          * the VLM refinement target becomes ``low_conf ∩ edge`` instead of
            all low-confidence pixels;
          * per-image VLM inference is skipped when an image has no
            low-confidence boundary pixels;
          * the confidence fusion boosts the VLM weight on edge pixels.

        Args:
            images:      [B, 3, H, W] unlabeled (weakly-augmented) inputs.
            pseudo_labels: [B, H, W] teacher argmax pseudo labels.
            conf_scores: [B, H, W] teacher max-softmax confidence.
            edge_map:    [B, H, W] bool CEE edge mask, or ``None``.
        """
        B, _, H, W = images.shape
        device = images.device

        self.current_iter += 1

        low_conf_mask = conf_scores < self.vlm_pp_conf_threshold

        # ---- CEE-guided target mask: only low-conf boundary pixels are VLM
        # targets. When CEE is off, priority == the original low_conf_mask,
        # which keeps the baseline behaviour bit-for-bit identical.
        use_cee_vlm = self.use_cee and self.use_cee_vlm and edge_map is not None
        if use_cee_vlm:
            if tuple(edge_map.shape) != tuple(pseudo_labels.shape):
                edge_map = F.interpolate(
                    edge_map.float().unsqueeze(1), size=(H, W), mode='nearest'
                ).squeeze(1).bool()
            priority_mask = low_conf_mask & edge_map
            # diagnostic counters (identical to the masks logged in semiearth.py)
            n_edge = int(edge_map.sum().item())
            n_low = int(low_conf_mask.sum().item())
            n_target = int(priority_mask.sum().item())
            if self.cee_debug and _is_rank0():
                print('[CEE-VLM DEBUG] edge_pixels={}, low_conf_pixels={}, '
                      'vlm_target_pixels={}'.format(n_edge, n_low, n_target))
            if n_target < self.min_edge_pixels:
                if self.cee_debug and _is_rank0():
                    print('[CEE-VLM] Skip VLM: target pixels {} < min_edge_pixels {}'.format(
                        n_target, self.min_edge_pixels))
                if self.use_vlm_on_mismatch:
                    return conf_scores, pseudo_labels
                return conf_scores
        else:
            priority_mask = low_conf_mask

        if not priority_mask.any():
            if self.cee_debug and _is_rank0():
                print('[CEE-VLM] Skip VLM inference: no low-confidence edge pixels')
            if self.use_vlm_on_mismatch:
                return conf_scores, pseudo_labels
            return conf_scores

        # Cache is only reused when explicitly enabled (baseline mode). Under
        # CEE-guided mode it is disabled by default because priority_mask and
        # pseudo labels change every iteration.
        should_infer = (self.current_iter % self.inference_interval == 0)
        if (not self.use_vlm_cache) or should_infer or self.cached_spatial_probs is None:
            all_spatial_probs = []

            for b in range(B):
                # skip VLM entirely for images without any VLM target region
                if not priority_mask[b].any():
                    all_spatial_probs.append(torch.zeros(self.num_classes, H, W))
                    continue

                pil_image = self._tensor_to_pil(images[b])
                spatial_probs = self._grounding_inference(pil_image, H, W)
                all_spatial_probs.append(spatial_probs)

            self.cached_spatial_probs = torch.stack(all_spatial_probs, dim=0).to(device)
            self.cached_shape = (B, H, W)

        qwen_spatial = self.cached_spatial_probs
        # safety: never reuse a cache with inconsistent batch / spatial shape
        if self.cached_shape != (B, H, W) or qwen_spatial.shape[0] != B:
            qwen_spatial = qwen_spatial[:B] if qwen_spatial.shape[0] > B else \
                F.pad(qwen_spatial, (0, 0, 0, 0, 0, 0, 0, B - qwen_spatial.shape[0]))

        vlm_pred_classes = qwen_spatial.argmax(dim=1)  # [B, H, W]
        vlm_max_conf = qwen_spatial.max(dim=1)[0]  # [B, H, W]

        qwen_confidence = torch.gather(
            qwen_spatial, 1, pseudo_labels.unsqueeze(1)
        ).squeeze(1)

        vlm_has_detection = vlm_max_conf > 0.1

        refined_conf = conf_scores.clone()
        refined_labels = pseudo_labels.clone() if self.use_vlm_on_mismatch else None

        # Only priority (CEE-guided) pixels are allowed to be modified by VLM.
        if self.use_vlm_on_mismatch:
            mismatch_mask = (pseudo_labels != vlm_pred_classes) & priority_mask & vlm_has_detection & (
            conf_scores < 0.3)
            if mismatch_mask.any():
                refined_labels[mismatch_mask] = vlm_pred_classes[mismatch_mask]
                refined_conf[mismatch_mask] = vlm_max_conf[mismatch_mask]

            consistent_mask = priority_mask & (~mismatch_mask) & vlm_has_detection
        else:
            consistent_mask = priority_mask & vlm_has_detection

        if consistent_mask.any():
            semi_weight = conf_scores[consistent_mask] / self.vlm_pp_conf_threshold
            qwen_weight = 1 - semi_weight

            # Edge-aware confidence refinement: boost the VLM influence on
            # semantic boundary pixels (no effect when CEE is off).
            if use_cee_vlm:
                edge_boost = torch.where(
                    edge_map[consistent_mask], self.edge_vlm_weight, 1.0
                )
                qwen_weight = torch.minimum(
                    qwen_weight * edge_boost, torch.ones_like(qwen_weight)
                )
                semi_weight = 1 - qwen_weight

            refined_conf[consistent_mask] = (
                semi_weight * conf_scores[consistent_mask] +
                qwen_weight * qwen_confidence[consistent_mask]
            )

        if self.use_vlm_on_mismatch:
            return refined_conf, refined_labels
        return refined_conf

QwenVLPurifiedSemiV1 = QwenVLPurifiedSemi

