import unittest

import numpy as np
import torch

from q2.data import prepare_split
from q2.missing import grid55, make_mask
from q2.model import (ConditionalRegressionHead, RAMP,
                      SignedConditionalRegressionHead)


class Q2ContractTest(unittest.TestCase):
    def sample(self):
        bert = np.zeros((2, 3, 50), dtype=np.int64)
        bert[:, 0, :5] = [101, 200, 201, 202, 102]
        bert[:, 1, :5] = 1
        audio = np.zeros((2, 50, 74), dtype=np.float32)
        vision = np.zeros((2, 50, 35), dtype=np.float32)
        audio[:, 1:4] = 1
        vision[:, 1:4] = 2
        return {"train": dict(id=["a", "b"], text_bert=bert, audio=audio, vision=vision,
                              classification_labels=np.array([0., 2.]),
                              regression_labels=np.array([-1., 1.]))}

    def test_field_major_loader(self):
        split = prepare_split(self.sample(), "train")
        self.assertEqual(split["text_bert"].shape, (2, 3, 50))
        self.assertEqual(split["raw_available"].shape, (2, 3, 50))
        self.assertEqual(split["raw_available"][0, 1].sum(), 3)
        self.assertFalse(split["content"][0, 0])
        self.assertFalse(split["content"][0, 4])

    def test_missing_masks_are_reproducible_and_content_only(self):
        content = prepare_split(self.sample(), "train")["content"]
        first = make_mask(content[0], (0, 1), .4, "random", 2, False, np.random.default_rng(7))
        second = make_mask(content[0], (0, 1), .4, "random", 2, False, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(first[:, ~content[0]].any())
        self.assertFalse(first[2].any())
        self.assertEqual(len(grid55(content)), 55)

    def test_corrupt_values_cannot_leak(self):
        split = prepare_split(self.sample(), "train")
        model = RAMP("M5").eval()
        bert = torch.tensor(split["text_bert"])
        audio = torch.tensor(split["audio"])
        vision = torch.tensor(split["vision"])
        content = torch.tensor(split["content"])
        available = torch.tensor(split["raw_available"])
        available[:, :, 2] = False
        with torch.inference_mode():
            reference = model(bert, audio, vision, content, available)["logits"]
            bert[:, 0, 2] = 29999
            audio[:, 2] = 1e6
            vision[:, 2] = -1e6
            changed = model(bert, audio, vision, content, available)["logits"]
        torch.testing.assert_close(reference, changed)

    def test_text_guided_residual_starts_from_baseline_and_masks_missing_av(self):
        split = prepare_split(self.sample(), "train")
        torch.manual_seed(7)
        baseline = RAMP("M4").eval()
        guided = RAMP("M4", fusion_mode="text_guided_residual").eval()
        guided.load_state_dict(baseline.state_dict(), strict=False)
        bert = torch.tensor(split["text_bert"])
        audio = torch.tensor(split["audio"])
        vision = torch.tensor(split["vision"])
        content = torch.tensor(split["content"])
        available = torch.tensor(split["raw_available"])
        with torch.inference_mode():
            base = baseline(bert, audio, vision, content, available)["logits"]
            initial = guided(bert, audio, vision, content, available)["logits"]
        torch.testing.assert_close(base, initial)
        with torch.no_grad():
            guided.av_residual[0][-1].bias.fill_(1.0)
        no_audio = available.clone()
        no_audio[:, 1] = False
        with torch.inference_mode():
            reference = guided(bert, audio, vision, content, no_audio)["logits"]
            corrupted = guided(bert, audio + 1e6, vision, content, no_audio)["logits"]
        torch.testing.assert_close(reference, corrupted)

    def test_hurdle_head_forms_three_class_distribution(self):
        split = prepare_split(self.sample(), "train")
        model = RAMP("M4", decision_head="hurdle").eval()
        with torch.inference_mode():
            logits = model(torch.tensor(split["text_bert"]),
                           torch.tensor(split["audio"]),
                           torch.tensor(split["vision"]),
                           torch.tensor(split["content"]),
                           torch.tensor(split["raw_available"]))["logits"]
        self.assertEqual(tuple(logits.shape), (2, 3))
        torch.testing.assert_close(logits.exp().sum(-1), torch.ones(2))

    def test_conditional_regression_head_preserves_linear_initial_prediction(self):
        torch.manual_seed(7)
        linear = torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, 1))
        conditional = ConditionalRegressionHead()
        conditional.mild.load_state_dict(linear.state_dict())
        conditional.strong.load_state_dict(linear.state_dict())
        x = torch.randn(5, 256)
        torch.testing.assert_close(conditional(x), linear(x))
        with torch.no_grad():
            conditional.strong[-1].bias.add_(1)
        prediction, mild, strong, gate_logit = conditional.components(x)
        torch.testing.assert_close(prediction, mild + gate_logit.sigmoid() * (strong - mild))

    def test_signed_conditional_head_mixes_three_experts(self):
        torch.manual_seed(11)
        linear = torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, 1))
        conditional = SignedConditionalRegressionHead()
        for expert in (conditional.negative, conditional.mild, conditional.positive):
            expert.load_state_dict(linear.state_dict())
        x = torch.randn(4, 256)
        torch.testing.assert_close(conditional(x), linear(x))
        with torch.no_grad():
            conditional.negative[-1].bias.sub_(1)
            conditional.positive[-1].bias.add_(1)
        prediction, experts, logits = conditional.components(x)
        torch.testing.assert_close(prediction, (logits.softmax(-1) * experts).sum(-1))



if __name__ == "__main__":
    unittest.main()
