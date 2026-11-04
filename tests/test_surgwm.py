import torch

from swm.baselines.surgwm import EMAVQ, LatentActionModel, MaskGITDynamics, SurgWMConfig

TINY = dict(height=16, width=24, n_codes=32, n_actions=4, dyn_dim=64, dyn_depth=2, dyn_heads=4)


def test_cached_frame_logits_match_full_forward():
    torch.manual_seed(0)
    c = SurgWMConfig(**TINY)
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


def test_window_states_match_a_forward_pass_on_the_crop():
    # generate() rebuilds the cache of frames lo..f-1 for each window, with no action into frame lo
    torch.manual_seed(0)
    c = SurgWMConfig(**TINY)
    model = MaskGITDynamics(c).eval()
    b, t, w, s = 2, 7, 4, (16 // 4) * (24 // 4)
    tokens = torch.randint(0, c.n_codes, (b, t, s))
    actions = torch.randint(0, c.n_actions, (b, t))
    with torch.no_grad():
        for f in range(w, t):
            lo = f + 1 - w
            acts = actions[:, lo:].clone()
            acts[:, 0] = c.n_actions
            past = [torch.zeros(b, w, s, c.dyn_dim) for _ in model.net.blocks]
            for j in range(lo, f):
                model.frame_logits(tokens[:, j], acts, j - lo, past)
            ref = model.logits(tokens[:, lo:f + 1], acts[:, :w])[:, -1]
            assert torch.allclose(model.frame_logits(tokens[:, f], acts, f - lo, past), ref, atol=1e-4)


def test_generate_sees_only_its_window():
    torch.manual_seed(0)
    c = SurgWMConfig(**TINY)
    model = MaskGITDynamics(c).eval()
    b, t, s = 2, 7, (16 // 4) * (24 // 4)
    tokens = torch.randint(0, c.n_codes, (b, t, s))
    actions = torch.randint(0, c.n_actions, (b, t))
    other = tokens.clone()
    other[:, :2] = torch.randint(0, c.n_codes, (b, 2, s))      # frames 0-1: outside the last frame's window of 5

    def run(x, window):
        return model.generate(x, actions, t - 1, steps=4, generator=torch.Generator().manual_seed(1), window=window)

    assert torch.equal(run(tokens, 5)[:, -1], run(other, 5)[:, -1])
    assert not torch.equal(run(tokens, None)[:, -1], run(other, None)[:, -1])
    assert torch.equal(run(tokens, t), run(tokens, None))


def test_ema_vq_follows_the_data_and_restarts_dead_codes():
    torch.manual_seed(0)
    vq = EMAVQ(8, 4).train()
    centres = torch.randn(4, 4) * 3 + 5                            # far from the initial codes
    h = (centres[torch.arange(256) % 4] + 0.1 * torch.randn(256, 4)).requires_grad_()
    _, before, _ = vq(h)
    restarted = vq.update()
    for _ in range(30):
        vq(h)
        restarted += vq.update()
    q, loss, idx = vq(h)
    loss.backward()
    assert restarted > 0                                           # codes far from the data restart on it
    assert loss < 0.01 * before                                    # every point has a code nearby
    assert len(idx.unique()) >= 4                                  # the four clusters each have their own code
    assert h.grad is not None and h.grad.abs().sum() > 0           # the commitment loss reaches the encoder


def test_action_model_is_causal_and_long_clips_use_training_length_windows():
    torch.manual_seed(0)
    c = SurgWMConfig(lam_height=8, lam_width=12, tok_dim=32, tok_heads=4, lam_enc_depth=1, lam_dec_depth=1, clip=4)
    lam = LatentActionModel(c).eval()
    x = torch.rand(2, 9, 3, 8, 12) * 2 - 1
    with torch.no_grad():
        idx, q, _ = lam.actions(x)
        assert idx.shape == (2, 8) and q.shape == (2, 8, c.code_dim)
        windows = [lam.actions(x[:, a:a + 4])[1] for a in (0, 3, 5)]   # transitions 0-2, 3-5, 5-7
        assert torch.allclose(q, torch.cat([windows[0], windows[1], windows[2][:, 1:]], 1), atol=1e-5)
        y = x.clone()
        y[:, 3:] = torch.rand_like(y[:, 3:])                          # frames after transition 1's frames 1-2
        assert torch.allclose(lam.actions(y[:, :4])[1][:, :2], windows[0][:, :2], atol=1e-5)
        rec, _, _ = lam(x[:, :4])
        assert torch.isfinite(rec)
