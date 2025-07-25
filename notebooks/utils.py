from ultralytics.nn.modules.head import Segment
import torch.nn as nn

def update_seg_head(model, nc=5):
    """
    Replaces the final YOLO Segment head with one configured for `nc` classes.
    Avoids reading any internal attributes from the existing Segment layer.
    """
    if hasattr(model.model, 'model'):
        seq = model.model.model  # This is the nn.Sequential container
        prev_layer = seq[-2]

        # Safely get number of output channels from the previous layer
        if hasattr(prev_layer, 'conv') and isinstance(prev_layer.conv, nn.Conv2d):
            ch = prev_layer.conv.out_channels
        elif isinstance(prev_layer, nn.Conv2d):
            ch = prev_layer.out_channels
        else:
            ch = 256  # fallback default

        # Replace the Segment head with a new one
        seq[-1] = Segment(ch=ch, nc=nc, n=3, act=True, c2f=True)
        print(f"[INFO] Replaced segmentation head: nc={nc}, ch={ch}")
    else:
        print("[ERROR] Model has no model.model structure.")
    
    return model
