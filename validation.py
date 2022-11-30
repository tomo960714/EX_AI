import torch
import numpy as np
from torch import nn
from constants import BATCH_SIZE
def validation_loop(valid_dataloader,model,loss_fn,device):
    pred_history = []
    true_history = []
    loss_history = []
    loss_tmp = 0
    nr_entries = 0
    model.eval()
    #https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-predict-new-samples-with-your-pytorch-model.md
    with torch.no_grad():
        batch_cnt = 0
        for batch, (input,target) in enumerate(valid_dataloader):
            input = torch.squeeze(input, dim=1)
            input = input.float()
            target=target.float()
            #print(input.shape)
            input=input.to(device)
            target=target.to(device)

            output = model(input)
            #_,preds = torch.max(output,1)
            loss = loss_fn(output,target)

            pred_history = pred_history + output.detach().cpu().numpy().tolist()
            true_history = true_history + target.detach().cpu().numpy().tolist()
            loss_tmp += loss.detach().cpu().numpy()
            nr_entries += len(input)
            #pred_reshape = torch.movedim(pred,1,0)
            #pred_new =nn.functional.softmax(pred_reshape[0],dim=0)
            
            #https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
        print(f"validation loss:{loss_tmp}, entries: {nr_entries}, avg: {loss_tmp/nr_entries}")

    
            
            
            
            
    #print('pred_history:',pred_history)
    #print('true_history:',true_history)

    return pred_history,true_history,loss_tmp/nr_entries
