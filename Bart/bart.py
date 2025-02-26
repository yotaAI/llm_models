import numpy as np
import math

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader


#Input Embedding
class InputEmbedding(nn.Module):
	def __init__(self,d_model:int,vocab_size:int):
		super().__init__()
		self.d_model=d_model
		self.embedding = nn.Embedding(vocab_size,d_model)
	def forward(self,x):
		return self.embedding(x) * math.sqrt(self.d_model)

# Positional Embedding
class PositionalEmbedding(nn.Module):
	def __init__(self,d_model:int,seq_len:int,dropout:float):
		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.dropout = nn.Dropout(dropout)

		pe = torch.zeros(seq_len,d_model)
		position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(-1)
		div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

		pe[:,0::2] = torch.sin(position / div_term)
		pe[:,1::2] = torch.cos(position / div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe',pe)
	def forward(self,x):
		x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
		return self.dropout(x)

# Attention Layer 
class MultiHeadAttenttionLayer(nn.Module):
	def __init__(self,d_model:int,h:int,dropout:float):
		super().__init__()
		self.d_model = d_model
		self.h = h
		assert d_model % h ==0, 'd_model should be divisible by h'

		self.d_k = d_model // h
		self.w_q = nn.Linear(d_model,d_model)
		self.w_k = nn.Linear(d_model,d_model)
		self.w_v = nn.Linear(d_model,d_model)
		self.w_o = nn.Linear(d_model,d_model)

		self.dropout = nn.Dropout(dropout)

	def attention(self,query,key,value,mask):
		d_k = query.shape[-1]
		attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
		if mask is not None:
			attention_score.mask_fill_(mask==0,-1e9)
		attention_score = self.dropout(attention_score)
		return (attention_score @ value) , attention_score

	def forward(self,q,k,v,mask):
		query = self.w_q(q)
		key   = self.w_k(k)
		value = self.w_v(v)

		query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(-2,-3)
		key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(-2,-3)
		value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(-2,-3)

		x,self.attention_score = self.attention(query,key,value,mask)

		x = x.transpose(-2,-3).contiguous()
		x = x.view(x.shape[0],x.shape[1],self.d_model)

		return self.w_o(x)

# Layer Normalization
class LayerNormalization(nn.Module):
	def __init__(self,eps:float=1e-6):
		super().__init__()
		self.eps = eps
		self.alpha = nn.Parameter(torch.ones(1))
		self.beta = nn.Parameter(torch.zeros(1))

	def forward(self,x):
		mean = x.mean(dim=-1,keepdims=True)
		std = x.std(dim=-1,keepdims=True)

		return self.alpha * (x - mean) / (std + self.eps) + self.beta
# Feed Forward
class FeedForward(nn.Module):
	def __init__(self,d_model:int,d_ff:int,dropout:float):
		super().__init__()
		self.linear_1 = nn.Linear(d_model,d_ff)
		self.linear_2 = nn.Linear(d_ff,d_model)
		self.dropout = nn.Dropout(dropout)
	def forward(self,x):
		return self.linear_2(self.dropout(self.linear_1(x)))

#Residual Connection
class ResidualConnection(nn.Module):
	def __init__(self,dropout:float):
		super().__init__()
		self.norm = LayerNormalization()
		self.dropout = nn.Dropout(dropout)
	
	def forward(self,x,residual):
		return self.dropout(self.norm(x)) + x

#Final Projection Layer
class ProjectionLayer(nn.Module):
	def __init__(self,d_model:int, vocab_size:int):
		super().__init__()
		self.d_model=d_model
		self.vocab_size=vocab_size

		self.proj = nn.Linear(d_model,vocab_size)
		# print(f'Vocab Size : {vocab_size}')
	def forward(self,x):
		return torch.log_softmax(self.proj(x),dim=-1)
# ------------------------------ Encoder------------------------
#Layer
class EncoderBlock(nn.Module):
	def __init__(self,self_attention_block:MultiHeadAttenttionLayer,feed_forward_block:FeedForward,dropout:float):
		super().__init__()
		self.self_attention_block = self_attention_block
		self.feed_forward_block = feed_forward_block
		self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
	def forward(self,x,src_mask):
		x = self.residual_connection[0](self.self_attention_block(x,x,x,src_mask),x)
		x = self.residual_connection[1](self.feed_forward_block(x),x)
		return x

class Encoder(nn.Module):
	def __init__(self,layers:nn.ModuleList):
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalization()
	def forward(self,x,mask):
		for layer in self.layers:
			x = layer(x,mask)
		return self.norm(x)

class BART(nn.Module):
	def __init__(self,encoder:Encoder,src_embeed:InputEmbedding,src_pos:PositionalEmbedding,projection_layer:ProjectionLayer):
		super().__init__()
		self.encoder = encoder
		self.src_embeed=src_embeed
		self.src_pos = src_pos
		self.projection_layer = projection_layer

	def encode(self,src,src_mask):
		src = self.src_embeed(src)
		src = self.src_pos(src)
		return self.encoder(src,src_mask)

	def project(self,x):
		return self.projection_layer(x)

def build_bart(vocab_size:int,seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
	embed = InputEmbedding(d_model,vocab_size)
	pos = PositionalEmbedding(d_model,seq_len,dropout)

	encoder_blocks = []
	for _ in range(N):
		self_attention_block = MultiHeadAttenttionLayer(d_model,h,dropout)
		feed_forward_block = FeedForward(d_model,d_ff,dropout)
		encoder_block = EncoderBlock(self_attention_block,feed_forward_block,dropout)
		encoder_blocks.append(encoder_block)

	encoder = Encoder(nn.ModuleList(encoder_blocks))

	projection_layer = ProjectionLayer(d_model,vocab_size)

	# Final Bart
	bart = BART(encoder,embed,pos,projection_layer)



	#Initialization of parameters
	for p in bart.parameters():
		if p.dim()>1:
			nn.init.xavier_uniform_(p)
	return bart



if __name__=='__main__':
	bart = build_bart(32000,100)
	print("Model Loaded")

	ip = torch.randint(1,31000,(1,100))
	a = bart.encode(ip,None)
	print(a.shape)
	b = bart.project(a)
	print(b.shape)