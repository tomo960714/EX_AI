from constants import NEPOCHS
import numpy as np
def validate(validation_dataloader,model,loss_fn):
    loss_history = []
    pred_history = []

    for iEpoch in range(NEPOCHS):
        valid_loss = 0.0
        for batch, (X,y) in enumerate(validation_dataloader):
            X=X.float()
            pred = model(X)
            loss = loss_fn(pred,y)
            #save pred
            pred_history.append(pred)


