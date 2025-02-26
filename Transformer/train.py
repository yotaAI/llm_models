from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Sequence, Whitespace, BertPreTokenizer
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

from datasets import BilingualDataset, causal_mask
from transformers import build_transformer
from config import get_config, get_weights_file_path

def get_all_sentences(ds,lang):
	for item in ds:
		yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang,vocab_size=10000):
	tokenizer_path= Path(config['tokenizer_file'].format(lang))
	if not Path.exists(tokenizer_path):
		tokenizer = Tokenizer(BPE(unk_token='<|UNK|>'))
		tokenizer.pre_tokenizer = Sequence([Whitespace(), BertPreTokenizer()])
		trainer = BpeTrainer(vocab_size=vocab_size,special_tokens=['<|UNK|>','<|CLS|>','<|SEP|>','<|PAD|>','<|MASK|>','<|SOS|>','<|EOS|>'],min_frequency=2)
		tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
		tokenizer.save(str(tokenizer_path))
	else:
		tokenizer =  Tokenizer.from_file(str(tokenizer_path))
	return tokenizer

def get_df(config):
	ds_raw = load_dataset('opus_books',f'{config['lang_src']}-{config['lang_tgt']}',split='train')

	#Build tokenizers
	tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
	tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

	#Keep 90% for Training and 10% for Validation
	train_ds_size = int(0.9 * len(ds_raw))
	val_ds_size = len(ds_raw) - train_ds_size
	train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])


	#Preparing datasets
	train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
	val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

	max_len_src = 0
	max_len_tgt = 0

	for item in ds_raw:
		src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
		tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

		max_len_src = max(max_len_src,len(src_ids))
		max_len_tgt = max(max_len_tgt,len(tgt_ids))

	print(f'Max length of source sentence : {max_len_src}')
	print(f'Max length of target sentence : {max_len_tgt}')

	train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
	val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

	return (train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt)



def get_model(config,vocab_size_len,vocab_tgt_len):
	model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
	return model



def train_model(config):
	device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device {device}')

	Path(config['model_folder']).mkdir(parents=True,exists_ok=True)

	train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_df(config)
	model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
	writer = SummaryWriter(config['experiment_name'])

	optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'], eps=1e-9)

	initial_epoch = 0
	global_step = 0
	if config['preload']:
		model_filename = get_weights_file_path(config,config['preload'])
		print(f'Preloading model {model_filename}')
		state = torch.load(model_filename)
		initial_epoch = state['epoch'] + 1
		optimizer.load_state_dict(state['optimizer_state_dict'])
		global_step = state['global_step']
		model.load_state_dict(state['model'])
	los_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("<|PAD|>"),label_smoothing=0.1).to(device)



	for epoch in range(initial_epoch,config['num_epochs']):
		model.train()
		batch_iterator = tqdm(train_dataloader,desc=f'Processing epoch {epoch:02d}')
		for batch in batch_iterator:
			encoder_input = batch['encoder_input'].to(device) #(batch,seqLen)
			decoder_input = batch['decoder_input'].to(device) #(batch,seqLen)
			encoder_mask = batch['encoder_mask'].to(device) #(batch,1,1,seqLen)
			decoder_mask = batch['decoder_Mask'].to(device) #(batch,1,seq_len,seqLen)



			#Run the tensors through the transformer
			encoder_output = model.encode(encoder_input,encoder_mask) #(B,seq_len,d_model)
			decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) #(B,seq_len,d_model)
			proj_output = model.project(decoder_output) #(B,seq_len,tgt_vocab_size)


			label =batch['label'].to(device) #(B,seq_len)

			loss = loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size(),label.view(-1))) #(B*seq_len,vocab_size) <> (B*seq_len)

			#Logging
			batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})
			writer.add_scalar('train loss', loss.item(),global_step)
			writer.flush()

			# Backpropagation & UPdate weights
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		# Save the model at the end
		model_filename = get_weights_file_path(config,f'{epoch:02d}')
		torch.save({
			'model': model.statedict(),
			'optimizer_state_dict': optimizer.statedict(),
			'global_step' : global_step,
			'epoch': epoch,
			},model_filename)


if __name__=='__main__':

	config = get_config()
	train_model(config)

