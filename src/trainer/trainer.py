from transformers import Trainer
import torch

def vicreg_loss(x1: torch.tensor, 
                x1_feat: torch.tensor, 
                y1: torch.tensor, 
                y1_feat: torch.tensor, args) -> torch.tensor:

    pass

class VICregTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        x = inputs.pop("x")
        y = inputs.pop("y")
        outputs = model(x, y)
        loss = vicreg_loss(outputs[0], outputs[1])
        return (loss, outputs) if return_outputs else loss