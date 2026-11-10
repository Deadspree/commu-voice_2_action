import torch


def motion_loss(prediction, target, mask, velocity_weight=0.1):
    # Both terms are in normalized angle units. Velocity uses displacement per
    # fixed-rate frame, so its coefficient is independent of clip length.
    angle = (prediction - target).abs()[mask].mean()
    valid = mask[:, 1:] & mask[:, :-1]
    if valid.any():
        velocity = ((prediction[:, 1:] - prediction[:, :-1]) -
                    (target[:, 1:] - target[:, :-1])).abs()[valid].mean()
    else:
        velocity = prediction.sum() * 0
    return angle + velocity_weight * velocity
