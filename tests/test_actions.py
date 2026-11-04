import torch

from swm.models.actions import hold_boxes, wrong_binding


def test_hold_boxes_fills_gaps_from_the_nearest_detection():
    nan = float("nan")
    b = torch.full((1, 5, 2, 4), nan)
    b[0, 1, 0] = torch.tensor([0.1, 0.1, 0.2, 0.2])
    b[0, 3, 0] = torch.tensor([0.5, 0.5, 0.6, 0.6])
    out = hold_boxes(b)
    assert torch.equal(out[0, :, 0, 0], torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5]))  # step 2 is a tie: the earlier
    assert torch.isnan(out[0, :, 1]).all()                                          # never detected: stays NaN


def test_wrong_binding_cycles_instrument_arms_only():
    occ = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4], [1, 5, 0, 0]])[:, None].expand(-1, 3, -1)   # (B, T, A)
    window = torch.tensor([[0, 7], [3, 1], [2, 2]])
    perm = wrong_binding(occ, window)
    inst = occ[:, 0] >= 2
    for b in range(3):
        assert sorted(perm[b].tolist()) == [0, 1, 2, 3]                                     # a permutation
        assert all(perm[b, a] == a for a in range(4) if not inst[b, a])                     # endoscope, empty: own
        assert all(inst[b, perm[b, a]] for a in range(4) if inst[b, a])                     # instruments swap only
    assert all(perm[b, a] != a for b in (0, 1) for a in range(4) if inst[b, a])             # two or more: none kept
    assert perm[2, 1] == 1                                                                  # one instrument: kept
    assert torch.equal(perm, wrong_binding(occ, window))                                    # fixed per window
