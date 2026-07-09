import torch
import numpy as np
from tqdm import tqdm
from transformers import Blip2ForImageTextRetrieval, AutoProcessor


class BLIP2ITMWrapper:
    def __init__(self, hf_id, device):
        self.model = Blip2ForImageTextRetrieval.from_pretrained(hf_id, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    def image_preprocess(self, pil_image):
        px = self.processor(images=pil_image, return_tensors="pt")["pixel_values"][0]
        return px

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Returns (t2i_scores, i2t_scores), each N x K x L — mirrors BLIPModelWrapper's contract."""
        all_scores = []
        for batch in tqdm(joint_loader, desc="Computing ITM scores"):
            pixel_values = batch["image_options"][0].to(self.device, torch.float16)  # B x C x H x W (K=1)
            B = pixel_values.shape[0]
            L = len(batch["caption_options"])

            batch_scores = np.zeros((B, 1, L), dtype=np.float32)
            for j, c_option in enumerate(batch["caption_options"]):
                text_inputs = self.processor.tokenizer(list(c_option), padding=True, truncation=True, return_tensors="pt").to(self.device)
                out = self.model(
                    pixel_values=pixel_values,
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    use_image_text_matching_head=True,
                )
                match_prob = out.logits_per_image.softmax(dim=-1)[:, 1]  # prob of "match"
                batch_scores[:, 0, j] = match_prob.float().cpu().numpy()

            all_scores.append(batch_scores)

        scores = np.concatenate(all_scores, axis=0)
        return scores, scores  # same for t2i/i2t here since we scored directly per-pair