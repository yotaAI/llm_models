import numpy as np
import math
import os
import sys
from pathlib import Path


import torch
import torch.nn as nn

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import Sequence, Whitespace, BertPreTokenizer
from torch.utils.tensorboard import SummaryWriter

#Local Module
from transformers import build_transformer
from config import get_config, get_weights_file_path

def get_all_sentences(ds,lang):
	for item in ds:
		yield item

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

def causal_mask(size):
	mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
	return mask == 0


if __name__ =='__main__':
	config = get_config()

	texts = ['Focus on one main topic.','Use supporting details to support the main idea.',
			'Conclude in a way that promotes the main idea.',
			'Use a capital letter at the beginning of every new sentence.',
			'Use a period at the end of each sentence.'
			]
	tokenizer = get_or_build_tokenizer(config,texts,'eng')
	print('Tokenizer Loaded')
	tokens = torch.tensor((tokenizer.encode(texts[0]).ids),dtype=torch.int64)
	max_len = 50
	remaining = max_len - len(tokens)
	tokens = torch.cat([
				tokens,
				torch.tensor([0]*remaining)
				]).unsqueeze(0)
	encoder_mask = (tokens!=0).unsqueeze(0).unsqueeze(0).int()
	decoder_input = torch.zeros(1,max_len).int()
	decoder_mask = (decoder_input!=0).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
	print(encoder_mask.shape,decoder_input.shape,decoder_mask.shape)
	model = build_transformer(src_vocab_size=61,
							tgt_vocab_size=61,
							src_seq_len=max_len,
							tgt_seq_len=max_len,
							d_model=128,
							N=2,
							h=8,
							)
	
	encoder_out = model.encode(tokens,encoder_mask)
	decoder_out = model.decode(encoder_out,encoder_mask,decoder_input,decoder_mask)
	proj_output = model.projection(decoder_out)
	print(tokenizer.decode(torch.argmax(proj_output,axis=-1)[0].tolist()))




