config = {
    "batch_size": 32,
    "epochs": 25,
    "seq_len": 256,
    "d_model": 256,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "pre_LN": True,
    "lang_src": "en",
    "lang_tgt": "it",
    "model_dir": "saved_model_states",
    "model_basename": "transfomers_model",
    "preload": None,
    "tokenizer_file": "Tokenizer_{0}.json",
    "experiments_name": "runs/transfomers_model",
    "train_size": 0.85,
    "val_size": 0.15
}