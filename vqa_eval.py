import argparse
import json
import os
import random
import re

import pandas as pd
import torch
from tqdm import tqdm

from dataset_zoo.aro_datasets import Controlled_Images, COCO_QA, VG_QA

LETTERS = ["A", "B", "C", "D"]


def get_dataset(name, root_dir="data"):
    if name == "Controlled_Images_A":
        return Controlled_Images(image_preprocess=None, root_dir=root_dir, subset="A", download=True)
    elif name == "Controlled_Images_B":
        return Controlled_Images(image_preprocess=None, root_dir=root_dir, subset="B", download=True)
    elif name == "COCO_QA_one_obj":
        return COCO_QA(image_preprocess=None, root_dir=root_dir, subset="one", download=True)
    elif name == "COCO_QA_two_obj":
        return COCO_QA(image_preprocess=None, root_dir=root_dir, subset="two", download=True)
    elif name == "VG_QA_one_obj":
        return VG_QA(image_preprocess=None, root_dir=root_dir, subset="one", download=True)
    elif name == "VG_QA_two_obj":
        return VG_QA(image_preprocess=None, root_dir=root_dir, subset="two", download=True)
    raise ValueError(name)


def build_prompt(options):
    lines = [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return (
        "Look at the image and choose the caption that correctly describes the spatial "
        "relationship between the objects shown.\n" + "\n".join(lines) +
        "\nRespond with only the letter of the correct option, nothing else."
    )


def shuffle_options(caption_options, seed):
    rng = random.Random(seed)
    idx = list(range(len(caption_options)))
    rng.shuffle(idx)
    shuffled = [caption_options[i] for i in idx]
    correct_letter = LETTERS[idx.index(0)]  # index 0 was always the correct caption
    return shuffled, correct_letter


def parse_answer(text):
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    return match.group(1) if match else None


class LlavaOnevisionScorer:
    def __init__(self, hf_id="llava-hf/llava-onevision-qwen2-7b-ov-hf", device="cuda"):
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch.float16, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    @torch.no_grad()
    def answer(self, image, prompt):
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
        chat_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=chat_prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return decoded


class Qwen25VLScorer:
    def __init__(self, hf_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch.float16, device_map=device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.device = device

    @torch.no_grad()
    def answer(self, image, prompt):
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        chat_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_prompt], images=[image], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return decoded
    
def get_scorer(model_name, device):
    if model_name == "llava-onevision":
        return LlavaOnevisionScorer(device=device)
    elif model_name == "qwen2.5-vl":
        return Qwen25VLScorer(device=device)
    raise ValueError(model_name)


def main(args):
    scorer = get_scorer(args.model_name, args.device)
    dataset = get_dataset(args.dataset)

    records = []
    for i in tqdm(range(len(dataset)), desc=f"{args.model_name} on {args.dataset}"):
        item = dataset[i]
        image = item["image_options"][0]
        shuffled_opts, correct_letter = shuffle_options(item["caption_options"], seed=i)
        prompt = build_prompt(shuffled_opts)

        raw_answer = scorer.answer(image, prompt)
        pred_letter = parse_answer(raw_answer)
        correct = (pred_letter == correct_letter)

        records.append({
            "Index": i, "Model": args.model_name, "Dataset": args.dataset,
            "Correct": correct, "Predicted": pred_letter, "GoldLetter": correct_letter,
            "RawOutput": raw_answer,
        })

    df = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"vqa_{args.model_name}_{args.dataset}.csv")
    df.to_csv(out_path, index=False)
    acc = df["Correct"].mean() * 100
    print(f"\n{args.model_name} on {args.dataset}: {acc:.1f}% ({df['Correct'].sum()}/{len(df)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=["llava-onevision", "qwen2.5-vl"])
    parser.add_argument("--dataset", required=True,
                         choices=["Controlled_Images_A", "Controlled_Images_B",
                                  "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="./outputs_vqa")
    args = parser.parse_args()
    main(args)