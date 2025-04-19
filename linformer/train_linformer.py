import os
import sys
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer,BartConfig,BartModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
from module import LinformerEncoder,BartDataset


#Hyperparameters.
layers=3
bsz=8
epoch=10
adam_eps=1e-5
weight_decay=0.01
learning_rate=1e-5
warmup=50
save_step=1000
save_pth='bart_distillition'
save_model_name='cnn_bart_encoder_'
dataset_pth='./cnn_dataset/'
print_step=20
os.makedirs(save_pth,exist_ok=True)


cnn_large_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
cnn_encoder = cnn_large_model.model.encoder.to(torch.bfloat16).to(device)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
print("Saving Large CNN Encoder Model ...")
torch.save(cnn_encoder.state_dict(),os.path.join(save_pth,'large_cnn.pt'))

lin_conf = BartConfig.from_pretrained("facebook/bart-large-cnn")
lin_conf._attn_implementation="eager"
lin_model_encoder = LinformerEncoder(lin_conf)
print("Saving Linformer CNN Encoder Model ...")
torch.save(lin_model_encoder.state_dict(),os.path.join(save_pth,'linformer_cnn.pt'))


train_df = pd.read_csv(os.path.join(dataset_pth,"train.csv")) 
train_ds = BartDataset(train_df,tokenizer,lin_conf)
train_loader = torch.utils.data.DataLoader(train_ds,batch_size=bsz,shuffle=True)
total_steps = epoch*len(train_loader)
print(f"Training for Total : {total_steps}")

del train_df


# loss_fn = torch.nn.KLDivLoss()

writer = SummaryWriter('log')

for layer in range(layers):
    lin_model = lin_model_encoder.layers[layer].to(torch.bfloat16).to(device)
    optimizer=torch.optim.Adam(lin_model.parameters(),lr=learning_rate,weight_decay=weight_decay,eps=adam_eps)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total_steps)
    loss_fn = torch.nn.L1Loss()

    total_loss=0
    for ep in range(epoch):
        lin_model.train(True)
        loader = tqdm.tqdm(train_loader)
        loader.set_description(f"Layer : {layer} | Epoch : {ep}")

        for idx,data in enumerate(loader):
            input_ids = data['input_ids'].to(device)
            with torch.no_grad():
                label_out = cnn_encoder(input_ids,attention_mask=None,output_hidden_states=True,output_attentions=True)
            dist_inp = label_out.hidden_states[layer]
            dist_out = label_out.hidden_states[layer + 1]
            
            optimizer.zero_grad()
            
            output = lin_model(hidden_states=dist_inp,attention_mask=None,layer_head_mask=None)[0]

            loss = loss_fn(dist_out,output)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss+=loss.item()
            loader.set_postfix({'Loss':loss.item()})
            if idx%print_step==0:
                writer.add_scalar(f"Loss/train/{layer}", loss.item(), ep*len(train_loader) + idx)
            if idx%save_step==0:
                torch.save(dict(
                    model=lin_model.state_dict(),
                    loss=total_loss/len(train_loader),
                    epoch=ep,
                    layer=layer,

                ),os.path.join(save_pth,f'tmp_lin_model_dist_l{layer}_e{ep}_idx{idx}.pt'))

        torch.save(dict(
            model=lin_model.state_dict(),
            loss=total_loss/len(train_loader),
            epoch=ep,
            layer=layer,

        ),os.path.join(save_pth,f'lin_model_dist_l{layer}.pt'))
            
