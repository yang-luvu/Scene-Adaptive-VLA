import re
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent))

PREDICTION_FILE_RE = re.compile(
    r"^language_preds_(?P<kind>[a-zA-Z0-9_]+)_rank_(?P<rank>\d+)(?:_.+)?\.json$"
)


def _load_prediction_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _select_latest_file_per_rank(files):
    latest = {}
    for file_path in files:
        match = PREDICTION_FILE_RE.match(file_path.name)
        if match is None:
            continue
        kind = match.group("kind")
        rank = int(match.group("rank"))
        key = (kind, rank)
        score = (file_path.stat().st_mtime_ns, file_path.name)
        if key not in latest or score > latest[key][0]:
            latest[key] = (score, file_path)
    ordered_keys = sorted(latest.keys(), key=lambda item: (item[0], item[1]))
    return [latest[key][1] for key in ordered_keys]


def resolve_prediction_files(root_path: str, pred_kind: str):
    pred_kind = {"commentary": "cot"}.get(pred_kind, pred_kind)
    root = Path(root_path)
    if root.is_file():
        return [root]

    if not root.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Prediction path must be a file or directory, got: {root}")

    pattern = f"language_preds_{pred_kind}_rank_*.json"
    candidates = sorted(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No prediction files matched {pattern} under {root}")
    return _select_latest_file_per_rank(candidates)


def load_predictions_from_paths(paths):
    merged = []
    for path in paths:
        merged.extend(_load_prediction_json(path))
    return merged


class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr", "METEOR", "SPICE"])
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}

    def eval_acc(self):
        scores = []
        for i in tqdm(range(len(self.accuracy["answer"]))):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
        # 仅在启用 ChatGPT 评分时加载可选依赖。
        from utils.gpt_eval import gpt_forward

        with Pool(16) as p:  # Change the number based on your CPU cores
            scores_all = p.map(gpt_forward, data)
        

        scores = [x for x in scores_all if x != -1]
        delted = len(scores_all) - len(scores)
        print(f"Deleted {delted} invalid samples")
        scores = list(map(float, scores))
        
        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        chunk_size = 500
        n_total = len(answer)

        # Fast path: single chunk
        if n_total <= chunk_size:
            results = self.language_eval.run_evaluation(answer, GT)
            return {f"val/{k}": float(v) for k, v in results.items()}

        # Split and combine afterwards (weighted by chunk size)
        results_accumulator = {}
        total_items = 0

        for i in range(0, n_total, chunk_size):
            answer_split = answer[i:i + chunk_size]
            GT_split = GT[i:i + chunk_size]
            chunk_len = len(answer_split)
            total_items += chunk_len

            results_gen = self.language_eval.run_evaluation(answer_split, GT_split)

            # Accumulate weighted sums
            for k, v in results_gen.items():
                results_accumulator[k] = results_accumulator.get(k, 0.0) + float(v) * chunk_len

        # Weighted mean over all items
        results_gen_dict = {f"val/{k}": v_sum / total_items for k, v_sum in results_accumulator.items()}
        return results_gen_dict


    def forward(self, answer, GT):
        answer = answer.replace('A: ', '')
        GT = GT.replace('A: ', '')
        answer = answer.replace(' <|im_end|>', '')
        GT = GT.replace(' <|im_end|>', '')
        self.accuracy["answer"].append(answer)
        self.accuracy["GT"].append(GT)
        self.GPT.append((answer, GT))
        self.language["GT"].append(GT)
        self.language["answer"].append(answer)
    def evaluation(self, skip_chatgpt=False):
        print("evaluation start!")
        scores = {}
        try:
            print("accuracy evaluation")
            scores["accuracy"] = self.eval_acc()
        except Exception:
            print("Error in accuracy evaluation")
            traceback.print_exc()
            scores["accuracy"] = 0.0
        if skip_chatgpt:
            scores["chatgpt"] = None
        else:
            try:
                print("chatGPT evaluation")
                scores["chatgpt"] = self.eval_chatGPT(self.GPT)
            except Exception:
                print("Error in chatGPT evaluation")
                traceback.print_exc()
                scores["chatgpt"] = 0.0
        try:
            print("language evaluation")
            scores["language"] = self.eval_language()
        except Exception:
            print("Error in language evaluation")
            traceback.print_exc()
            scores["language"] = {}

        return scores

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, default="outputs/simlingo/predictions/language_preds_cot_rank_0.json", help='path to prediction file or prediction directory')
    parser.add_argument('--pred_kind', type=str, default="qa", choices=["qa", "cot", "all", "commentary"], help='prediction file kind when root_path1 is a directory')
    parser.add_argument('--skip-chatgpt', action='store_true', help='Skip OpenAI/ChatGPT scoring and compute exact/language metrics only.')
    args = parser.parse_args()

    prediction_files = resolve_prediction_files(args.root_path1, args.pred_kind)
    print("Using prediction files:")
    for path in prediction_files:
        print(f"  - {path}")

    pred_file = load_predictions_from_paths(prediction_files)
        
    gt = [preds[1].replace('A: ','') for preds in pred_file]
    pred = [preds[0].replace('A: ','') for preds in pred_file]

    print("Number of predictions: ", len(pred))
    print("Number of ground truths: ", len(gt))
    
    evaluation = evaluation_suit()
    
    for i in range(len(pred)):
        evaluation.forward(pred[i], gt[i])

    output = evaluation.evaluation(skip_chatgpt=args.skip_chatgpt)

    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("language score: ", output["language"])
    
    # save the evaluation results
    root_path = Path(args.root_path1)
    if root_path.is_dir():
        save_path = root_path / f"language_preds_{args.pred_kind}_merged_metrics_gpt-4o-2024-08-06.json"
    else:
        save_path = Path(str(root_path).replace(".json", "_metrics_gpt-4o-2024-08-06.json"))
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)
