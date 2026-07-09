import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

class HFDualEncoderWrapper:
    """Generic wrapper for HF dual-encoder VLMs that expose
    get_image_features() / get_text_features() — works for SigLIP2 and MetaCLIP2."""

    def __init__(self, hf_id, device, dtype=torch.float16):
        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device
        self.dtype = dtype

    def image_preprocess(self, pil_image):
        # Returns a single preprocessed tensor [C,H,W]; the DataLoader stacks these into batches.
        px = self.processor(images=pil_image, return_tensors="pt")["pixel_values"][0]
        return px.to(self.dtype)

    @staticmethod
    def _extract(output):
        """get_image_features/get_text_features return a BaseModelOutputWithPooling
        (or occasionally a plain tuple) rather than a bare tensor in current transformers."""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if isinstance(output, tuple):
            return output[1] if len(output) > 1 else output[0]
        return output  # already a tensor, older transformers behavior

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        scores = []
        for batch in tqdm(joint_loader, desc="Computing retrieval scores"):
            image_options = []
            for i_option in batch["image_options"]:
                out = self.model.get_image_features(pixel_values=i_option.to(self.device, self.dtype))
                img_feats = self._extract(out)
                img_feats = F.normalize(img_feats, dim=-1).float().cpu().numpy()
                image_options.append(np.expand_dims(img_feats, axis=1))

            caption_options = []
            for c_option in batch["caption_options"]:
                text_inputs = self.processor(text=list(c_option), padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self.model.get_text_features(**text_inputs)
                txt_feats = self._extract(out)
                txt_feats = F.normalize(txt_feats, dim=-1).float().cpu().numpy()
                caption_options.append(np.expand_dims(txt_feats, axis=1))

            image_options = np.concatenate(image_options, axis=1)
            caption_options = np.concatenate(caption_options, axis=1)
            scores.append(np.einsum("nkd,nld->nkl", image_options, caption_options))

        return np.concatenate(scores, axis=0)