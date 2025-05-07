import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.utils.data import Dataset, DataLoader

class DPORewardModel(nn.Module):
	def __init__(self,base_model):
		self.model=base_model
		self.reward_head = nn.Sequential(
			nn.Linear(self.model.config.hidden_size,256),
			nn.Linear(256,1),
			)
	def forward(self,input_ids,attention_mask=None):
		output = self.model(input_ids=input_ids,attention_mask=attention_mask)
		hidden_state = output['last_hidden_state'][:,0]
		reward = self.reward_head(hidden_state)
		return reward

class DPORewardLoss(nn.Module):
	def __init__(self):
		self.log_sig_loss = nn.LogSigmoid()
	def forward(self,reward_choosen,reward_rejected):
		return -self.log_sig_loss(reward_choosen - reward_rejected).mean()


class RewardDataset(Dataset):
	def __init__(self,dataset,tokenizer=None,max_seq_len=None):
		self.dataset=dataset
		self.tokenizer=tokenizer
		self.max_seq_len=max_seq_len
		if max_seq_len==None and tokenizer!=None:
			raise Exception("Max Sequence Length is missing.")

	def __len__(self):
		return len(self.datset)

	def __getitem__(self,idx):
		data = self.dataset.iloc[idx]
		prompt = data['prompt']
		selected = data['selected']
		rejected = data['rejected']

		if self.tokenizer:
			prompt = tokenizer(prompt,seq_len=self.max_seq_len,padding='seq_len',truncation=True,return_tensors='pt')
			selected = tokenizer(selected,seq_len=self.max_seq_len,padding='seq_len',truncation=True,return_tensors='pt')
			rejected = tokenizer(rejected,seq_len=self.max_seq_len,padding='seq_len',truncation=True,return_tensors='pt')

			return dict(
					prompt_ids=prompt['input_ids'].unsqueeze(0),
					prompt_mask=prompt['attention_mask'].unsqueeze(0),
					selected_ids=selected['input_ids'].unsqueeze(0),
					selected_mask=selected['attention_mask'].unsqueeze(0),
					rejected_ids=rejected['input_ids'].unsqueeze(0),
					rejected_mask=rejected['attention_mask'].unsqueeze(0),
				)

		return dict(
				prompt=prompt,
				selected=selected,
				rejected=rejected,
			)


# Training
train_datset = RewardDataset() # Add dataset

model = DPORewardModel() # Add the base model

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = DPORewardLoss()

for batch in dataloader:  # Your DataLoader yields: prompt, chosen, rejected
    prompt, chosen, rejected = batch

    # Tokenize
    chosen_inputs = tokenizer(prompt + chosen, return_tensors="pt", padding=True, truncation=True)
    rejected_inputs = tokenizer(prompt + rejected, return_tensors="pt", padding=True, truncation=True)

    # Forward pass
    reward_chosen = model(**chosen_inputs)
    reward_rejected = model(**rejected_inputs)

    # Compute loss
    loss = loss_fn(reward_chosen, reward_rejected)

    # Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()












