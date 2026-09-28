"""Q2 audit, training, validation grid and frozen attachment-3 inference."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import RobustAVScaler, audit, content_mask, load_aligned
from .missing import grid55, mixed_masks
from .model import RAMP


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(split: dict, batch_size: int, order: np.ndarray | None = None):
    if order is None:
        order = np.arange(len(split["id"]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield idx, {k: split[k][idx] for k in ("text_bert", "audio", "vision", "content", "raw_available", "label_cls", "label_reg")}


def to_torch(batch: dict, drop: np.ndarray, device: str):
    available = batch["raw_available"] & ~drop & batch["content"][:, None]
    return dict(text_bert=torch.as_tensor(batch["text_bert"], dtype=torch.long, device=device),
                audio=torch.as_tensor(batch["audio"], dtype=torch.float32, device=device),
                vision=torch.as_tensor(batch["vision"], dtype=torch.float32, device=device),
                content=torch.as_tensor(batch["content"], dtype=torch.bool, device=device),
                available=torch.as_tensor(available, dtype=torch.bool, device=device))


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in range(3):
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(scores))


def evaluate(model: RAMP, split: dict, drop: np.ndarray, device: str, batch_size: int = 64):
    model.eval()
    probs, scores = [], []
    with torch.inference_mode():
        for idx, batch in batches(split, batch_size):
            out = model(**to_torch(batch, drop[idx], device))
            probs.append(out["logits"].softmax(-1).cpu().numpy())
            scores.append(out["pred_reg"].clamp(-3, 3).cpu().numpy())
    probs, scores = np.concatenate(probs), np.concatenate(scores)
    pred = probs.argmax(1)
    actual = split["label_cls"]
    reg = split["label_reg"]
    pearson = float(np.corrcoef(reg, scores)[0, 1]) if np.std(scores) and np.std(reg) else float("nan")
    metrics = dict(accuracy=float(np.mean(pred == actual)), macro_f1=macro_f1(actual, pred),
                   mae=float(np.mean(np.abs(scores - reg))), pearson=pearson)
    return metrics, probs, scores


def selection(clean: dict, mini: list[dict]) -> float:
    return (.5 * clean["macro_f1"] + .5 * np.mean([m["macro_f1"] for m in mini])
            - .5 * clean["mae"] / 3 - .5 * np.mean([m["mae"] for m in mini]) / 3)


def supervised_contrastive(pooled: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Single-view supervised contrastive loss on the fused representation."""
    features = F.normalize(pooled.float(), dim=-1)
    similarities = features @ features.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    logits = similarities.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return pooled.sum() * 0
    return -(log_prob.masked_fill(~positives, 0).sum(dim=1)[valid] / counts[valid]).mean()


def balanced_order(labels: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    by_class = [np.flatnonzero(labels == cls) for cls in range(3)]
    if any(len(indices) == 0 for indices in by_class):
        raise ValueError("balanced batches require all three classes")
    n_batches = int(np.ceil(len(labels) / batch_size))
    chunks = []
    for _ in range(n_batches):
        counts = np.array([batch_size // 3] * 3)
        counts[:batch_size % 3] += 1
        indices = np.concatenate([rng.choice(by_class[cls], int(counts[cls]), replace=True)
                                  for cls in range(3)])
        rng.shuffle(indices)
        chunks.append(indices)
    return np.concatenate(chunks)


def train(args):
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.output) / args.variant / str(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    data = load_aligned(args.data)
    scaler = RobustAVScaler().fit(data["train"])
    scaler.save(output / "scaler.npz")
    train_split = scaler.transform(data["train"])
    valid = scaler.transform(data["valid"])
    model = RAMP(args.variant, text_encoder_type=args.text_encoder, bert_path=args.bert,
                 neutral_aux=args.neutral_aux_weight > 0,
                 fusion_mode=args.fusion_mode, decision_head=args.decision_head,
                 bert_train_last_n=args.bert_train_last_n).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)["state"]
        if args.decision_head == "hurdle":
            initial = {k: v for k, v in initial.items() if not k.startswith("cls.")}
        missing, unexpected = model.load_state_dict(initial, strict=False)
        if unexpected or any(not (k.startswith(("bert.", "av_residual.", "av_reliability.",
                                                "neutral_gate.", "polarity_gate.")) or k == "modal_logits") for k in missing):
            raise ValueError(f"initial checkpoint mismatch: {missing[:5]} {unexpected[:5]}")
    if args.text_pretrain:
        if args.text_encoder != "compact":
            raise ValueError("text distillation initialization applies only to compact encoder")
        pretrained = torch.load(args.text_pretrain, map_location="cpu", weights_only=False)["state"]
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        if unexpected or any(k.startswith(("word.", "pos.", "text_context.")) for k in missing):
            raise ValueError("text distillation checkpoint does not match model")
    if args.residual_only:
        if args.fusion_mode != "text_guided_residual" or not args.init_checkpoint:
            raise ValueError("residual-only requires text-guided fusion and an initial checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("av_residual.", "av_reliability.")))
    if args.head_only:
        if args.decision_head != "hurdle" or not args.init_checkpoint or args.residual_only:
            raise ValueError("head-only requires a hurdle head and an initial checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("neutral_gate.", "polarity_gate.")))
    if args.modal_aux_weight and (args.variant != "M4" or args.residual_only or args.head_only):
        raise ValueError("modal auxiliary training requires ordinary M4 training")
    modal_aux = (torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, 3))
        for _ in range(3)]).to(device) if args.modal_aux_weight else None)
    parameter_groups = [
        {"params": [p for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith("bert.")]
                   + (list(modal_aux.parameters()) if modal_aux is not None else []), "lr": args.lr},
        {"params": [p for name, p in model.named_parameters()
                    if p.requires_grad and name.startswith("bert.")], "lr": args.bert_lr},
    ]
    optimizer = torch.optim.AdamW([group for group in parameter_groups if group["params"]],
                                  weight_decay=1e-4)
    class_weight = None
    class_log_count = None
    class_margin = None
    if args.class_loss == "balanced_softmax" and args.class_weight:
        raise ValueError("balanced softmax and class weighting cannot be combined")
    if args.class_loss == "balanced_softmax":
        counts = np.bincount(train_split["label_cls"], minlength=3)
        class_log_count = torch.as_tensor(np.log(counts), dtype=torch.float32, device=device)
    if args.class_weight:
        counts = np.bincount(train_split["label_cls"], minlength=3)
        values = np.sqrt(counts.sum() / (3 * counts))
        values /= values.mean()
        class_weight = torch.as_tensor(values, dtype=torch.float32, device=device)
    if args.class_loss == "ldam_drw":
        counts = np.bincount(train_split["label_cls"], minlength=3)
        margins = counts.astype(np.float64) ** (-0.25)
        margins *= args.ldam_max_margin / margins.max()
        class_margin = torch.as_tensor(margins, dtype=torch.float32, device=device)
    if args.class_loss == "adaptive_soft" and args.label_smoothing:
        raise ValueError("adaptive soft labels and fixed label smoothing cannot be combined")
    rng = np.random.default_rng(args.seed)
    mask_set = grid55(valid["content"])
    mini_ids = ("T_10_front", "T_25_middle", "T_40_end", "A_25_middle",
                "V_25_middle", "A+V_25_middle")
    best, no_gain = -1e9, 0
    history = []
    for epoch in range(args.epochs):
        if args.variant == "M5" and epoch == args.warmup:
            # Stage A is a convergence check; choose deployable checkpoints
            # only after Stage B has actually seen missing-modality batches.
            best, no_gain = -1e9, 0
        model.train()
        if modal_aux is not None:
            modal_aux.train()
        order = (balanced_order(train_split["label_cls"], args.batch_size, rng)
                 if args.balanced_batch else rng.permutation(len(train_split["id"])))
        total_loss = 0.0
        modal_confidences, modal_modulations = [], []
        for _, batch in batches(train_split, args.batch_size, order):
            corrupt = args.variant == "M5" and epoch >= args.warmup
            drop = mixed_masks(batch["content"], rng) if corrupt else np.zeros_like(batch["raw_available"])
            feed = to_torch(batch, drop, device)
            target_cls = torch.as_tensor(batch["label_cls"], dtype=torch.long, device=device)
            target_reg = torch.as_tensor(batch["label_reg"], dtype=torch.float32, device=device)
            def objective():
                out = model(**feed)
                class_logits = out["logits"] + class_log_count if class_log_count is not None else out["logits"]
                if args.class_loss == "ldam_drw":
                    # Defer both the class-dependent margin and loss reweighting.
                    if epoch >= args.drw_start_epoch:
                        class_logits = class_logits.scatter_add(
                            1, target_cls[:, None], -class_margin[target_cls, None])
                        class_loss = F.cross_entropy(class_logits, target_cls, weight=class_weight)
                    else:
                        class_loss = F.cross_entropy(class_logits, target_cls)
                elif args.class_loss == "adaptive_soft":
                    epsilon = args.adaptive_epsilon * torch.exp(-target_reg.abs() / args.adaptive_tau)
                    soft_target = F.one_hot(target_cls, 3).float() * (1 - epsilon[:, None]) + epsilon[:, None] / 3
                    class_loss = F.cross_entropy(class_logits, soft_target, weight=class_weight)
                else:
                    class_loss = F.cross_entropy(class_logits, target_cls, weight=class_weight,
                                                 label_smoothing=args.label_smoothing)
                regression_loss = (F.l1_loss(out["pred_reg"], target_reg)
                                   if args.reg_loss == "l1" else
                                   F.smooth_l1_loss(out["pred_reg"], target_reg,
                                                    beta=args.reg_beta))
                loss = class_loss + regression_loss
                if args.supcon_weight > 0:
                    loss = loss + args.supcon_weight * supervised_contrastive(
                        out["pooled"], target_cls, args.supcon_temperature)
                if args.neutral_aux_weight > 0:
                    neutral_target = (target_cls == 1).float()
                    loss = loss + args.neutral_aux_weight * F.binary_cross_entropy_with_logits(
                        out["neutral_logit"], neutral_target)
                modal_confidence = None
                if modal_aux is not None:
                    modal_logits = [head(out["modal_features"][:, i])
                                    for i, head in enumerate(modal_aux)]
                    modal_loss = torch.stack([
                        F.cross_entropy(logits, target_cls, weight=class_weight)
                        for logits in modal_logits]).mean()
                    loss = loss + args.modal_aux_weight * modal_loss
                    modal_confidence = torch.stack([
                        logits.detach().softmax(-1).gather(1, target_cls[:, None]).mean()
                        for logits in modal_logits])
                return loss, modal_confidence

            optimizer.zero_grad(set_to_none=True)
            loss, modal_confidence = objective()
            loss.backward()
            if args.sam_rho > 0:
                active = [parameter for parameter in model.parameters()
                          if parameter.requires_grad and parameter.grad is not None]
                grad_norm = torch.linalg.vector_norm(torch.stack(
                    [torch.linalg.vector_norm(parameter.grad) for parameter in active]))
                scale = args.sam_rho / (grad_norm + 1e-12)
                perturbations = []
                with torch.no_grad():
                    for parameter in active:
                        perturbation = parameter.grad * scale
                        parameter.add_(perturbation)
                        perturbations.append(perturbation)
                optimizer.zero_grad(set_to_none=True)
                objective()[0].backward()
                with torch.no_grad():
                    for parameter, perturbation in zip(active, perturbations):
                        parameter.sub_(perturbation)
            modulation = None
            if modal_confidence is not None and epoch >= args.ogm_warmup_epochs:
                ratios = modal_confidence / modal_confidence.mean().clamp_min(1e-8)
                modulation = (1 - torch.tanh(args.ogm_alpha * (ratios - 1).clamp_min(0))).tolist()
            if modal_confidence is not None:
                modal_confidences.append(modal_confidence.tolist())
                modal_modulations.append(modulation if modulation is not None else [1.0] * 3)
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if name.startswith(("audio.", "vision.", "av_context.")):
                    branch = 1 if name.startswith(("audio.", "av_context.0.")) else 2
                    parameter.grad.mul_(args.av_grad_scale * (modulation[branch] if modulation else 1))
                elif name.startswith("text_projection."):
                    parameter.grad.mul_(args.text_grad_scale * (modulation[0] if modulation else 1))
            torch.nn.utils.clip_grad_norm_(list(model.parameters())
                                           + (list(modal_aux.parameters()) if modal_aux is not None else []), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target_cls)
        clean, _, _ = evaluate(model, valid, mask_set["clean"], device)
        mini = [evaluate(model, valid, mask_set[k], device)[0] for k in mini_ids]
        score = selection(clean, mini)
        eligible = args.variant != "M5" or epoch >= args.warmup
        record = dict(epoch=epoch + 1, loss=total_loss / len(order),
                      selection=score, checkpoint_eligible=eligible,
                      clean=clean, mini={k: v for k, v in zip(mini_ids, mini)})
        if modal_confidences:
            record["modal_confidence"] = np.mean(modal_confidences, axis=0).tolist()
            record["modal_modulation"] = np.mean(modal_modulations, axis=0).tolist()
        history.append(record)
        print(json.dumps(record), flush=True)
        if eligible and score > best + 1e-5:
            best, no_gain = score, 0
            trainable_bert = {k for k, p in model.named_parameters()
                              if k.startswith("bert.") and p.requires_grad}
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("bert.") or k in trainable_bert}
            torch.save(dict(state=state, variant=args.variant, seed=args.seed,
                            epoch=epoch + 1, selection=score, class_weight=args.class_weight,
                            label_smoothing=args.label_smoothing, lr=args.lr,
                            text_pretrain=args.text_pretrain, init_checkpoint=args.init_checkpoint,
                            text_encoder=args.text_encoder,
                            neutral_aux_weight=args.neutral_aux_weight,
                            fusion_mode=args.fusion_mode,
                            residual_only=args.residual_only,
                            decision_head=args.decision_head,
                            head_only=args.head_only,
                            bert_train_last_n=args.bert_train_last_n,
                            bert_lr=args.bert_lr,
                            class_loss=args.class_loss,
                            ldam_max_margin=args.ldam_max_margin,
                            drw_start_epoch=args.drw_start_epoch,
                            adaptive_epsilon=args.adaptive_epsilon,
                            adaptive_tau=args.adaptive_tau,
                            reg_loss=args.reg_loss,
                            reg_beta=args.reg_beta,
                            supcon_weight=args.supcon_weight,
                            supcon_temperature=args.supcon_temperature,
                            balanced_batch=args.balanced_batch,
                            av_grad_scale=args.av_grad_scale,
                            text_grad_scale=args.text_grad_scale,
                            sam_rho=args.sam_rho,
                            modal_aux_weight=args.modal_aux_weight,
                            ogm_alpha=args.ogm_alpha,
                            ogm_warmup_epochs=args.ogm_warmup_epochs,
                            modal_aux_state=(modal_aux.state_dict() if modal_aux is not None else None),
                            bert_path=args.bert), output / "best.pt")
        elif eligible:
            no_gain += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
        if epoch >= args.warmup and no_gain >= args.patience:
            break
    print(f"best={best:.5f} path={output / 'best.pt'}", flush=True)


def load_frozen(checkpoint: Path, device: str, bert_path: str | None = None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = saved.get("text_encoder", "compact")
    model = RAMP(saved["variant"], text_encoder_type=encoder,
                 bert_path=bert_path or saved.get("bert_path"),
                 neutral_aux=saved.get("neutral_aux_weight", 0) > 0,
                 fusion_mode=saved.get("fusion_mode", "weighted"),
                 decision_head=saved.get("decision_head", "flat"),
                 bert_train_last_n=saved.get("bert_train_last_n", 0),
                 reg_head_type=saved.get("reg_head_type", "linear")).to(device)
    missing, unexpected = model.load_state_dict(saved["state"], strict=False)
    if unexpected or any(not (k.startswith("bert.") or k == "modal_logits") for k in missing):
        raise ValueError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()
    scaler = RobustAVScaler.load(checkpoint.parent / "scaler.npz")
    return model, scaler


def grid(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(args.checkpoint)
    model, scaler = load_frozen(checkpoint, device, args.bert)
    split = scaler.transform(load_aligned(args.data)[args.split])
    masks = grid55(split["content"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition_id", "sample_id", "true_cls", "true_reg", "pred_cls", "pred_reg",
                         "prob_negative", "prob_neutral", "prob_positive"])
        for condition, drop in masks.items():
            metric, probs, scores = evaluate(model, split, drop, device)
            metric_rows.append({"condition_id": condition, **metric})
            for i, sample_id in enumerate(split["id"]):
                writer.writerow([condition, sample_id, int(split["label_cls"][i]),
                                 float(split["label_reg"][i]), int(probs[i].argmax()), float(scores[i]),
                                 *map(float, probs[i])])
            print(condition, metric, flush=True)
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)


def attachment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler = load_frozen(Path(args.checkpoint), device, args.bert)
    files = sorted(Path(args.input).glob("附件3_*.pkl"))
    if len(files) != 30:
        raise ValueError(f"expected 30 attachment-3 files, got {len(files)}")
    rows, diagnostics = [], []
    def block_stats(mask):
        runs, current, longest = 0, 0, 0
        for value in mask:
            if value:
                if current == 0:
                    runs += 1
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return runs, longest

    for path in files:
        with path.open("rb") as f:
            source = pickle.load(f)["test"]
        if not all(len(source[k]) == 1 for k in ("text_bert", "audio", "vision")):
            raise ValueError(f"{path.name}: expected field-major singleton")
        bert = np.asarray(source["text_bert"][0], dtype=np.int64)[None]
        audio = np.asarray(source["audio"][0], dtype=np.float32)[None]
        vision = np.asarray(source["vision"][0], dtype=np.float32)[None]
        content = content_mask(bert)
        # Attachment 3 encodes removed words as [UNK]=100; match a shared A/V gap
        # to avoid confusing a naturally unknown word with synthetic missingness.
        zero_a = ~np.any(audio != 0, 2)
        zero_v = ~np.any(vision != 0, 2)
        missing_t = content & (bert[:, 0] == 100) & zero_a & zero_v
        raw = np.stack((content & ~missing_t, content & ~zero_a, content & ~zero_v), 1)
        split = dict(text_bert=bert, audio=audio, vision=vision, content=content,
                     raw_available=raw)
        split = scaler.transform(split)
        feed = to_torch(split, np.zeros_like(raw), device)
        with torch.inference_mode():
            out = model(**feed)
            prob = out["logits"].softmax(-1)[0].cpu().numpy()
            score = float(out["pred_reg"].clamp(-3, 3)[0].cpu())
        number = int(path.stem.split("_")[-1])
        rows.append(dict(sample_no=number, pred_label=("Negative", "Neutral", "Positive")[int(prob.argmax())],
                         pred_intensity=score, prob_negative=float(prob[0]),
                         prob_neutral=float(prob[1]), prob_positive=float(prob[2])))
        missing = content[:, None] & ~raw
        stats = [block_stats(missing[0, m]) for m in range(3)]
        diagnostics.append(dict(sample_no=number,
                                text_missing_ratio=float(missing[0, 0].sum() / max(1, content.sum())),
                                audio_missing_ratio=float(missing[0, 1].sum() / max(1, content.sum())),
                                vision_missing_ratio=float(missing[0, 2].sum() / max(1, content.sum())),
                                text_missing_blocks=stats[0][0], audio_missing_blocks=stats[1][0],
                                vision_missing_blocks=stats[2][0], max_gap_text=stats[0][1],
                                max_gap_audio=stats[1][1], max_gap_vision=stats[2][1],
                                missing_modalities="+".join(k for m, k in enumerate(("T", "A", "V")) if missing[0, m].any()),
                                pred_entropy=float(-(prob * np.log(np.clip(prob, 1e-12, 1))).sum()),
                                max_probability=float(prob.max())))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("attachment3_predictions.csv", rows), ("attachment3_missing_diagnostics.csv", diagnostics)):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if sorted(row["sample_no"] for row in rows) != list(range(1, 31)):
        raise ValueError("sample numbering invalid")
    for row in rows:
        probs = np.array([row["prob_negative"], row["prob_neutral"], row["prob_positive"]])
        if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1, atol=1e-5):
            raise ValueError(f"sample {row['sample_no']}: invalid probabilities")
        if not np.isfinite(row["pred_intensity"]) or not -3 <= row["pred_intensity"] <= 3:
            raise ValueError(f"sample {row['sample_no']}: invalid intensity")
    print(f"wrote {len(rows)} attachment-3 predictions to {output}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=[f"M{x}" for x in range(6)], required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--class-weight", action="store_true")
    p.add_argument("--class-loss", choices=("cross_entropy", "balanced_softmax", "ldam_drw", "adaptive_soft"),
                   default="cross_entropy")
    p.add_argument("--ldam-max-margin", type=float, default=0.3)
    p.add_argument("--drw-start-epoch", type=int, default=2,
                   help="Zero-based epoch when LDAM and class weighting start")
    p.add_argument("--adaptive-epsilon", type=float, default=0.1)
    p.add_argument("--adaptive-tau", type=float, default=0.5)
    p.add_argument("--supcon-weight", type=float, default=0.0)
    p.add_argument("--supcon-temperature", type=float, default=0.1)
    p.add_argument("--balanced-batch", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--reg-loss", choices=("smooth_l1", "l1"), default="smooth_l1")
    p.add_argument("--reg-beta", type=float, default=1.0,
                   help="Smooth L1 transition point; ignored for L1")
    p.add_argument("--text-pretrain")
    p.add_argument("--init-checkpoint")
    p.add_argument("--text-encoder", choices=["compact", "bert_base"], default="compact")
    p.add_argument("--bert", help="Local bert-base-uncased directory")
    p.add_argument("--neutral-aux-weight", type=float, default=0.0)
    p.add_argument("--fusion-mode", choices=("weighted", "text_guided_residual"),
                   default="weighted")
    p.add_argument("--residual-only", action="store_true")
    p.add_argument("--decision-head", choices=("flat", "hurdle"), default="flat")
    p.add_argument("--head-only", action="store_true")
    p.add_argument("--bert-train-last-n", type=int, default=0)
    p.add_argument("--bert-lr", type=float, default=1e-5)
    p.add_argument("--av-grad-scale", type=float, default=1.0)
    p.add_argument("--text-grad-scale", type=float, default=1.0)
    p.add_argument("--sam-rho", type=float, default=0.0)
    p.add_argument("--modal-aux-weight", type=float, default=0.0)
    p.add_argument("--ogm-alpha", type=float, default=1.0)
    p.add_argument("--ogm-warmup-epochs", type=int, default=1)
    p = sub.add_parser("grid")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bert")
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p = sub.add_parser("attachment3")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bert")
    args = parser.parse_args()
    if args.command == "train" and (args.reg_beta <= 0 or args.adaptive_tau <= 0
                                      or args.supcon_temperature <= 0
                                      or args.supcon_weight < 0 or args.adaptive_epsilon < 0
                                      or args.adaptive_epsilon > 1
                                      or args.av_grad_scale <= 0 or args.text_grad_scale <= 0
                                      or args.sam_rho < 0 or args.modal_aux_weight < 0
                                      or args.ogm_alpha < 0 or args.ogm_warmup_epochs < 0):
        parser.error("adaptive and contrastive hyperparameters must be in range")
    if args.command == "audit":
        _, report = audit(args.data, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        train(args)
    elif args.command == "grid":
        grid(args)
    else:
        attachment(args)


if __name__ == "__main__":
    main()
