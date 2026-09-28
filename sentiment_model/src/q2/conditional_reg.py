"""Retrain a two-regime regression head on a frozen, weighted M4 backbone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.nn import functional as F

from .model import ConditionalRegressionHead, SignedConditionalRegressionHead
from .feature_cache import MINI, create_cache
from .run import load_frozen, seed_all


def validation(head, cache, device, threshold):
    head.eval()
    target = cache["valid_reg"].to(device)
    with torch.inference_mode():
        mae = {}
        for name, pooled in cache["valid_pooled"].items():
            prediction = head(pooled.to(device)).squeeze(-1).clamp(-3, 3)
            mae[name] = float((prediction - target).abs().mean())
        if isinstance(head, ConditionalRegressionHead):
            _, _, _, gate_logit = head.components(cache["valid_pooled"]["clean"].to(device))
            strong = (target.abs() > threshold).float()
            gate = gate_logit.sigmoid()
            diagnostics = dict(gate_brier=float(((gate - strong) ** 2).mean()),
                               mean_gate_strong=float(gate[strong.bool()].mean()),
                               mean_gate_other=float(gate[~strong.bool()].mean()))
        else:
            _, _, logits = head.components(cache["valid_pooled"]["clean"].to(device))
            states = torch.where(target < -threshold, 0,
                                 torch.where(target > threshold, 2, 1))
            diagnostics = dict(gate_accuracy=float((logits.argmax(-1) == states).float().mean()),
                               mean_true_state_probability=float(logits.softmax(-1).gather(
                                   1, states[:, None]).mean()))
    score = 0.5 * mae["clean"] + 0.5 * np.mean([mae[name] for name in MINI])
    return float(score), mae, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--head-type", choices=("binary", "signed"), default="binary")
    parser.add_argument("--strong-threshold", type=float, default=1.5)
    parser.add_argument("--gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--expert-loss-weight", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if (args.strong_threshold <= 0 or args.strong_threshold >= 3 or
            args.gate_loss_weight < 0 or args.expert_loss_weight < 0 or
            args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0 or args.lr <= 0):
        parser.error("invalid training hyperparameters")
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    copyfile(args.base_checkpoint.parent / "scaler.npz", args.output / "scaler.npz")
    if args.cache.exists():
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        if cache["base_checkpoint"] != str(args.base_checkpoint.resolve()):
            raise ValueError("cache belongs to another checkpoint")
    else:
        cache = create_cache(args.base_checkpoint, args.data, args.bert, args.cache)

    base_model, _ = load_frozen(args.base_checkpoint, device, str(args.bert))
    if (base_model.variant != "M4" or base_model.decision_head != "flat" or
            base_model.text_encoder_type != "bert_base" or base_model.bert_train_last_n or
            base_model.reg_head_type != "linear"):
        raise ValueError("requires original flat M4 with frozen bert-base-uncased")
    original = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    train_pooled = cache["train_pooled"]
    train_reg = cache["train_reg"]
    if args.head_type == "binary":
        head = ConditionalRegressionHead().to(device)
        head.mild.load_state_dict(base_model.reg.state_dict())
        head.strong.load_state_dict(base_model.reg.state_dict())
        prior = float((train_reg.abs() > args.strong_threshold).float().mean())
        with torch.no_grad():
            head.strength_gate[-1].weight.zero_()
            head.strength_gate[-1].bias.fill_(float(np.log(prior / (1 - prior))))
    else:
        head = SignedConditionalRegressionHead().to(device)
        for expert in (head.negative, head.mild, head.positive):
            expert.load_state_dict(base_model.reg.state_dict())
        labels = train_reg.numpy()
        states = np.where(labels < -args.strong_threshold, 0,
                          np.where(labels > args.strong_threshold, 2, 1))
        priors = np.bincount(states, minlength=3) / len(states)
        if np.any(priors == 0):
            raise ValueError("all three signed strength states need training examples")
        gate_class_weight = torch.as_tensor((1 / np.sqrt(priors)) /
                                             (1 / np.sqrt(priors)).mean(),
                                             dtype=torch.float32, device=device)
        with torch.no_grad():
            head.state_gate[-1].weight.zero_()
            head.state_gate[-1].bias.copy_(torch.as_tensor(np.log(priors),
                                                           dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, initial_mae, initial_gate = validation(head, cache, device, args.strong_threshold)
    history = [dict(epoch=0, validation_score=best,
                    validation_mae=initial_mae, gate=initial_gate)]

    def save(epoch, score):
        saved = dict(original)
        saved["state"] = {name: value for name, value in original["state"].items()
                          if not name.startswith("reg.")}
        saved["state"].update({f"reg.{name}": value.detach().cpu().clone()
                               for name, value in head.state_dict().items()})
        saved.update(reg_head_type=("conditional" if args.head_type == "binary"
                                    else "signed_conditional"),
                     conditional_reg_epoch=epoch,
                     conditional_reg_seed=args.seed,
                     conditional_reg_head_type=args.head_type,
                     conditional_reg_base=str(args.base_checkpoint.resolve()),
                     conditional_reg_threshold=args.strong_threshold,
                     conditional_reg_gate_loss_weight=args.gate_loss_weight,
                     conditional_reg_expert_loss_weight=args.expert_loss_weight,
                     conditional_reg_selection=score)
        torch.save(saved, args.output / "best.pt")

    save(0, best)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = rng.permutation(len(train_pooled))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            target = train_reg[idx].to(device)
            if args.head_type == "binary":
                strong = (target.abs() > args.strong_threshold).float()
                prediction, mild_pred, strong_pred, gate_logit = head.components(
                    train_pooled[idx].to(device))
                gate_loss = F.binary_cross_entropy_with_logits(gate_logit, strong)
                mild_mask, strong_mask = ~strong.bool(), strong.bool()
                expert_loss = 0.0
                if mild_mask.any():
                    expert_loss = expert_loss + F.l1_loss(mild_pred[mild_mask], target[mild_mask])
                if strong_mask.any():
                    expert_loss = expert_loss + F.l1_loss(strong_pred[strong_mask], target[strong_mask])
                expert_loss = expert_loss / 2
            else:
                state = torch.where(target < -args.strong_threshold, 0,
                                    torch.where(target > args.strong_threshold, 2, 1))
                prediction, experts, gate_logits = head.components(
                    train_pooled[idx].to(device))
                gate_loss = F.cross_entropy(gate_logits, state, weight=gate_class_weight)
                losses = []
                for state_id in range(3):
                    mask = state == state_id
                    if mask.any():
                        losses.append(F.l1_loss(experts[mask, state_id], target[mask]))
                expert_loss = sum(losses) / len(losses)
            main_loss = F.l1_loss(prediction, target)
            loss = (main_loss + args.gate_loss_weight * gate_loss +
                    args.expert_loss_weight * expert_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        score, valid_mae, gate_diag = validation(head, cache, device,
                                                  args.strong_threshold)
        row = dict(epoch=epoch, train_loss=total / len(order),
                   validation_score=score, validation_mae=valid_mae,
                   gate=gate_diag)
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best - 1e-5:
            best, stale = score, 0
            save(epoch, score)
        else:
            stale += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2),
                                                     encoding="utf-8")
        if stale >= args.patience:
            break
    print(json.dumps(dict(baseline=history[0]["validation_score"], best=best,
                          checkpoint=str(args.output / "best.pt"))), flush=True)


if __name__ == "__main__":
    main()
