from pathlib import Path

def get_config():
  return {
    "model_1": {
    "batch_size": 32,
    "num_epochs": 150,
    "lr": 10**-4,
    "seq_len": 128,
    "d_model": 64,
    "N": 2,
    "h" : 4,
    "lang_src": "indonesian",
    "lang_tgt": "english",
    "model_folder": "weights/model_1",
    "model_basename": "tmodel_",
    "preload": "149",
    "tokenizer_file":"./mt/tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
  },
  "model_2": {
    "batch_size": 32,
    "num_epochs": 150,
    "lr": 10**-4,
    "seq_len": 128,
    "d_model": 64,
    "N": 4,
    "h" : 4,
    "lang_src": "indonesian",
    "lang_tgt": "english",
    "model_folder": "weights/model_2",
    "model_basename": "tmodel_",
    "preload": "119",
    "tokenizer_file":"./mt/tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
  },
  "model_3": {
    "batch_size": 32,
    "num_epochs": 150,
    "lr": 10**-4,
    "seq_len": 128,
    "d_model": 64,
    "N": 4,
    "h" : 8,
    "lang_src": "indonesian",
    "lang_tgt": "english",
    "model_folder": "weights/model_3",
    "model_basename": "tmodel_",
    "preload": "99",
    "tokenizer_file":"./mt/tokenizer_{0}.json",
    "experiment_name": "runs/tmodel"
  }
  }

def get_weights_file_path(config, epoch: str):
  model_folder = config["model_folder"]
  model_basename = config["model_basename"]
  model_filename = f"{model_basename}{epoch}.pt"
  return str(Path(".") / model_folder / model_filename)