import torch

from swm.models.actions import hold_boxes


def test_hold_boxes_fills_gaps_from_the_nearest_detection():
    nan = float("nan")
    b = torch.full((1, 5, 2, 4), nan)
    b[0, 1, 0] = torch.tensor([0.1, 0.1, 0.2, 0.2])
    b[0, 3, 0] = torch.tensor([0.5, 0.5, 0.6, 0.6])
    out = hold_boxes(b)
    assert torch.equal(out[0, :, 0, 0], torch.tensor([0.1, 0.1, 0.1, 0.5, 0.5]))  # step 2 is a tie: the earlier
    assert torch.isnan(out[0, :, 1]).all()                                          # never detected: stays NaN
