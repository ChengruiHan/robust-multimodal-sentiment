"""Q2 backbone variants; the final M4 ensemble is named STATE-MSA.

Missing text is replaced before token encoding. RAMP stays as the Python class
name so existing checkpoints and experiment code remain compatible.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ObservedAttention(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = .1):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.null_key = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_value = nn.Parameter(torch.zeros(1, 1, dim))
        self.relative = nn.Embedding(101, heads)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, observed: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        def heads(t):
            return t.view(b, -1, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        nk = heads(self.null_key.expand(b, -1, -1))
        nv = heads(self.null_value.expand(b, -1, -1))
        k, v = torch.cat((k, nk), 2), torch.cat((v, nv), 2)
        score = q @ k.transpose(-1, -2) / math.sqrt(dim // self.heads)
        pos = torch.arange(length, device=x.device)
        distance = (pos[:, None] - pos[None, :] + 50).clamp(0, 100)
        bias = self.relative(distance).permute(2, 0, 1)
        score[:, :, :, :length] += bias[None]
        key_mask = torch.cat((observed, ~observed.any(1, keepdim=True)), 1)
        score = score.masked_fill(~key_mask[:, None, None, :], -1e4)
        attn = score.softmax(-1)
        context = (attn @ v).transpose(1, 2).reshape(b, length, dim)
        x = x + self.dropout(self.out(context))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x.masked_fill(~content[:, :, None], 0)


class ConditionalRegressionHead(nn.Module):
    """Two small experts routed by a strength gate over the shared M4 representation."""

    def __init__(self, dim: int = 256, hidden: int = 32):
        super().__init__()
        self.mild = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.strong = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.strength_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                           nn.GELU(), nn.Linear(hidden, 1))

    def components(self, pooled: torch.Tensor):
        mild = self.mild(pooled).squeeze(-1)
        strong = self.strong(pooled).squeeze(-1)
        gate_logit = self.strength_gate(pooled).squeeze(-1)
        gate = gate_logit.sigmoid()
        return (1 - gate) * mild + gate * strong, mild, strong, gate_logit

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.components(pooled)[0].unsqueeze(-1)


class SignedConditionalRegressionHead(nn.Module):
    """Separate negative and positive strong tails with one learned three-way gate."""

    def __init__(self, dim: int = 256, hidden: int = 32):
        super().__init__()
        self.negative = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.mild = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.positive = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.state_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                                        nn.GELU(), nn.Linear(hidden, 3))

    def components(self, pooled: torch.Tensor):
        experts = torch.cat((self.negative(pooled), self.mild(pooled),
                             self.positive(pooled)), dim=-1)
        logits = self.state_gate(pooled)
        prediction = (logits.softmax(-1) * experts).sum(-1)
        return prediction, experts, logits

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.components(pooled)[0].unsqueeze(-1)


class RAMP(nn.Module):
    def __init__(self, variant: str = "M5", dim: int = 128, vocab: int = 30522,
                 text_encoder_type: str = "compact", bert_path: str | None = None,
                 neutral_aux: bool = False, fusion_mode: str = "weighted",
                 decision_head: str = "flat", bert_train_last_n: int = 0,
                 reg_head_type: str = "linear"):
        super().__init__()
        if variant not in {f"M{x}" for x in range(6)}:
            raise ValueError(variant)
        self.variant = variant
        self.text_encoder_type = text_encoder_type
        self.has_neutral_aux = neutral_aux
        if fusion_mode not in {"weighted", "text_guided_residual"}:
            raise ValueError(fusion_mode)
        self.fusion_mode = fusion_mode
        if decision_head not in {"flat", "mlp", "hurdle"}:
            raise ValueError(decision_head)
        self.decision_head = decision_head
        if reg_head_type not in {"linear", "conditional", "signed_conditional"}:
            raise ValueError(reg_head_type)
        self.reg_head_type = reg_head_type
        if not 0 <= bert_train_last_n <= 12:
            raise ValueError(bert_train_last_n)
        self.bert_train_last_n = bert_train_last_n
        if text_encoder_type == "compact":
            self.word = nn.Embedding(vocab, dim, padding_idx=0)
            self.text_context = nn.ModuleList([ObservedAttention(dim) for _ in range(2)])
        elif text_encoder_type == "bert_base":
            if not bert_path:
                raise ValueError("bert_path is required for bert_base")
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
            self.bert.requires_grad_(False).eval()
            for layer in self.bert.encoder.layer[-bert_train_last_n:] if bert_train_last_n else ():
                layer.requires_grad_(True)
            self.text_projection = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim))
            # Start from a reliable text anchor; learn whether A/V add value.
            self.modal_logits = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))
        else:
            raise ValueError(text_encoder_type)
        if text_encoder_type != "bert_base" and bert_train_last_n:
            raise ValueError("BERT layer tuning requires bert_base")
        self.pos = nn.Embedding(51, dim)
        self.audio = nn.Linear(74, dim)
        self.vision = nn.Linear(35, dim)
        self.av_context = nn.ModuleList([nn.ModuleList([ObservedAttention(dim) for _ in range(2)]) for _ in range(2)])
        self.modality = nn.Parameter(torch.randn(3, dim) * .02)
        self.state = nn.Embedding(2, dim)
        self.missing = nn.Parameter(torch.zeros(3, dim))
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh(), nn.Linear(dim, 1))
        if fusion_mode == "text_guided_residual":
            self.av_residual = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim),
                              nn.GELU(), nn.Linear(dim, dim)) for _ in range(2)
            ])
            self.av_reliability = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, 1)) for _ in range(2)
            ])
            for branch in self.av_residual:
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)
        layer = nn.TransformerEncoderLayer(dim, 4, 256, .1, batch_first=True, norm_first=True)
        self.global_context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.emo = nn.Parameter(torch.randn(1, 1, dim) * .02)
        if decision_head == "flat":
            self.cls = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 3))
        elif decision_head == "mlp":
            self.cls = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 64),
                                     nn.GELU(), nn.Dropout(.1), nn.Linear(64, 3))
        else:
            self.neutral_gate = nn.Sequential(nn.LayerNorm(dim * 2),
                                              nn.Linear(dim * 2, 64), nn.GELU(),
                                              nn.Dropout(.1), nn.Linear(64, 1))
            self.polarity_gate = nn.Sequential(nn.LayerNorm(dim * 2),
                                               nn.Linear(dim * 2, 64), nn.GELU(),
                                               nn.Dropout(.1), nn.Linear(64, 1))
        if reg_head_type == "linear":
            self.reg = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 1))
        elif reg_head_type == "conditional":
            self.reg = ConditionalRegressionHead(dim * 2)
        else:
            self.reg = SignedConditionalRegressionHead(dim * 2)
        if neutral_aux:
            self.neutral_aux = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.text_encoder_type == "bert_base":
            self.bert.train(mode and self.bert_train_last_n > 0)
        return self

    def forward(self, text_bert, audio, vision, content, available):
        batch, length = content.shape
        pos = self.pos(torch.arange(length, device=content.device))[None]
        ids = text_bert[:, 0].clone()
        # This is the only entry point for synthetic text corruption.
        ids = torch.where(content & ~available[:, 0], 103, ids)
        if self.text_encoder_type == "bert_base":
            if self.bert_train_last_n:
                hidden = self.bert(input_ids=ids, attention_mask=text_bert[:, 1],
                                   token_type_ids=text_bert[:, 2]).last_hidden_state
            else:
                with torch.no_grad():
                    hidden = self.bert(input_ids=ids, attention_mask=text_bert[:, 1],
                                       token_type_ids=text_bert[:, 2]).last_hidden_state
            text = self.text_projection(hidden)
            text = text.masked_fill(~content[:, :, None], 0)
        else:
            text = self.word(ids) + pos
            text_observed = available[:, 0] & content
            for layer in self.text_context:
                text = layer(text, text_observed, content)
        values = [text]
        for m, (input_tensor, projection) in enumerate(((audio, self.audio), (vision, self.vision)), 1):
            observed = available[:, m] & content
            value = projection(input_tensor)
            if self.variant >= "M2":
                value = torch.where(observed[:, :, None], value, self.missing[m])
            else:
                value = value * observed[:, :, None]
            if self.variant >= "M3":
                for layer in self.av_context[m - 1]:
                    value = layer(value + pos, observed, content)
            values.append(value)
        modal_means = []
        for modality, value in enumerate(values):
            observed = available[:, modality] & content
            count = observed.sum(1).clamp_min(1)[:, None]
            modal_means.append((value * observed[:, :, None]).sum(1) / count)
        modal_features = torch.stack(modal_means, 1)
        stack = torch.stack(values, 2)
        state = self.state(available.long().transpose(1, 2))
        stack = stack + self.modality[None, None] + pos[:, :, None]
        if self.variant >= "M4":
            stack = stack + state
            scores = self.gate(torch.cat((stack, state), -1)).squeeze(-1)
            if self.text_encoder_type == "bert_base":
                scores = scores + self.modal_logits[None, None]
            if self.variant == "M0":
                scores[:, :, 1:] = -1e4
            weights = scores.softmax(-1)
        elif self.variant == "M0":
            weights = torch.zeros(batch, length, 3, device=stack.device)
            weights[:, :, 0] = 1
        else:
            if self.text_encoder_type == "bert_base":
                weights = self.modal_logits.softmax(0)[None, None].expand(batch, length, -1)
            else:
                weights = torch.full((batch, length, 3), 1 / 3, device=stack.device)
        if self.variant == "M0":
            weights = torch.zeros_like(weights)
            weights[:, :, 0] = 1
        fused = (stack * weights[:, :, :, None]).sum(2)
        if self.fusion_mode == "text_guided_residual":
            # A/V may correct an observed text token, but missing A/V cannot
            # contribute through projection biases or learned missing tokens.
            text_observed = available[:, 0] & content
            for modality in (1, 2):
                context = torch.cat((text, values[modality],
                                     text * values[modality]), -1)
                correction = self.av_residual[modality - 1](context)
                reliability = torch.sigmoid(self.av_reliability[modality - 1](context))
                observed = text_observed & available[:, modality]
                fused = fused + .5 * correction * reliability * observed[:, :, None]
        fused = fused.masked_fill(~content[:, :, None], 0)
        sequence = torch.cat((self.emo.expand(batch, -1, -1), fused), 1)
        pad = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=content.device), ~content), 1)
        encoded = self.global_context(sequence, src_key_padding_mask=pad)
        mean = (fused * content[:, :, None]).sum(1) / content.sum(1).clamp_min(1)[:, None]
        pooled = torch.cat((encoded[:, 0], mean), -1)
        if self.decision_head in {"flat", "mlp"}:
            logits = self.cls(pooled)
        else:
            neutral = self.neutral_gate(pooled)
            polarity = self.polarity_gate(pooled)
            logits = torch.cat((nn.functional.logsigmoid(-neutral) + nn.functional.logsigmoid(-polarity),
                                nn.functional.logsigmoid(neutral),
                                nn.functional.logsigmoid(-neutral) + nn.functional.logsigmoid(polarity)), -1)
        result = {"logits": logits, "pred_reg": self.reg(pooled).squeeze(-1),
                  "modal_weights": weights, "pooled": pooled,
                  "modal_features": modal_features}
        if self.has_neutral_aux:
            result["neutral_logit"] = self.neutral_aux(pooled).squeeze(-1)
        return result
