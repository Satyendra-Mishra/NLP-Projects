import os, math
from pathlib import Path
import numpy as np
import tokenizers
import datasets
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from src.utils import get_dataset, save_model_states
from src.tokenization import build_tokenizer, get_tokenizers, get_all_sentences
from src.dataset import build_dataset, build_dataloader
from src.models import build_transformer
from src.train import train_model, run_validation


if __name__ == "__main__":
    
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get the raw translation dataset
    raw_data = get_dataset(config, subset="train")
    # buold or get the tokenizers
    tokenizer_src, tokenizer_tgt = get_tokenizers(config, raw_data)
    # pytorch datasets
    train_dataset, val_dataset = build_dataset(config, raw_data, tokenizer_src, tokenizer_tgt)
    # build dataloaders
    train_dataloader = build_dataloader(train_dataset, config)
    val_dataloader = build_dataloader(val_dataset, config)
    # source and target vocab sizes    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    # instantiate the transformers model
    transformer = build_transformer(src_vocab_size, 
                                    tgt_vocab_size, 
                                    seq_len = config["seq_len"], 
                                    d_model = config["d_model"],
                                    h = config["num_heads"],
                                    L_enc = config["num_encoder_layers"],
                                    L_dec = config["num_decoder_layers"],
                                    d_ff = 2048, 
                                    norm_first = config["pre_LN"])
    model = transformer.to(device)
    # Tensorboard summary writer
    summary_writer = SummaryWriter(log_dir=config["experiments_name"])
    # optimizer
    optimizer = torch.optim.Adam(transformer.parameters(), lr = 10e-4, eps=1e-9)
    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    # train the model
    train_losses, val_losses = train_model(config=config,
                                           model = transformer,
                                           train_loader=train_dataloader,
                                           val_loader=val_dataloader,
                                           optim=optimizer,
                                           loss_fn=criterion,
                                           device=device,
                                           summary_writer=summary_writer,
                                           src_tokenizer=tokenizer_src,
                                           tgt_tokenizer=tokenizer_tgt)