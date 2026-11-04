import torch

from swm.baselines.surgwm import MaskGITDynamics, SurgWMConfig


def test_cached_frame_logits_match_full_forward():
    torch.manual_seed(0)
    c = SurgWMConfig(height=16, width=24, n_codes=32, n_actions=4, dyn_dim=64, dyn_depth=2, dyn_heads=4)
    model = MaskGITDynamics(c).eval()
    b, t, s = 2, 5, (16 // 4) * (24 // 4)
    tokens = torch.randint(0, c.n_codes, (b, t, s))
    actions = torch.randint(0, c.n_actions + 1, (b, t))
    past = [torch.zeros(b, t, s, c.dyn_dim) for _ in model.net.blocks]
    with torch.no_grad():
        full = model.logits(tokens, actions)
        for f in range(t):
            frame = tokens[:, f].clone()
            frame[:, ::3] = model.mask_id                       # a partly decoded frame, as during MaskGIT
            ref = model.logits(torch.cat([tokens[:, :f], frame[:, None]], 1), actions[:, :f + 1])[:, f]
            assert torch.allclose(model.frame_logits(frame, actions, f, past), ref, atol=1e-4)
            assert torch.allclose(model.frame_logits(tokens[:, f], actions, f, past), full[:, f], atol=1e-4)
