from monai.losses.dice import DiceLoss, DiceCELoss

def get_loss(loss_type="dice_ce"):
    if loss_type == "dice":
        return DiceLoss(
            smooth_nr=0, smooth_dr=1e-5,
            squared_pred=True, to_onehot_y=False, sigmoid=True
        )
    elif loss_type == "dice_ce":
        return DiceCELoss(
            smooth_nr=0, smooth_dr=1e-5,
            squared_pred=True, to_onehot_y=False, sigmoid=True
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")