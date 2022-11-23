import torch
import numpy as np
from torch import nn
from constants import BATCH_SIZE
def validate(valid_dataloader,model,loss_fn,device):
    pred_history = []
    true_history = []
    model.eval()
    #https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-predict-new-samples-with-your-pytorch-model.md
    with torch.no_grad():
        batch_cnt = 0
        for batch, (input,target) in enumerate(valid_dataloader):
            
            input = input.float()
            target=target.float()
            #print(input.shape)
            input=input.to(device)
            target=target.to(device)

            pred = model(input)
            #pred_reshape = torch.movedim(pred,1,0)
            #pred_new =nn.functional.softmax(pred_reshape[0],dim=0)
            
            #https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
            tmp_pred=pred.detach().cpu().numpy()
            tmp_true =target.detach().cpu().numpy()
            
            for i in range(len(tmp_pred)):
                pred_history.append(tmp_pred[i,])
                true_history.append(tmp_true[i,])
            batch_cnt += 1
            
            
            
    #print('pred_history:',pred_history)
    #print('true_history:',true_history)

    return pred_history,true_history
