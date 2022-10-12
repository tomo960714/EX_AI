import torch
import numpy as np
def validate(valid_dataloader,model):
    
    ##https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-predict-new-samples-with-your-pytorch-model.md
    with torch.no_grad():
        for batch, (X,y) in enumerate(valid_dataloader):
            X =X.float()
            pred = model(X)
            pred_class =np.argmax(pred)

            print(f'prediction: {pred:>4f}, class: {pred_class:>4f}, original: {y:>4f}')
