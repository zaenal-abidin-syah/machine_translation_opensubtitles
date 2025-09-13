from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace




import re
import string

from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
# from config import get_config, get_weights_file_path
# from dataset import BilingualDataset, causal_mask

from datasets import Dataset
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import math
from evaluate import load
from .model import build_transformer
from .dataset import BilingualDataset, causal_mask
from .config import get_config, get_weights_file_path
# def get_all_sentences(ds, lang):
#   for item in ds:
#     yield item['translation'][lang]

def get_all_sentences_2(ds, lang):
  for item in ds:
    yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config["tokenizer_file"].format(lang)).expanduser().resolve()
  print("path tokenizer : ", tokenizer_path)
  if not Path.exists(tokenizer_path):
    print("tokenizer file tidak ada?")
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences_2(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    print("tokenizer file ada")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

def remove_emoji(text):
  emoji_pattern = re.compile(
      "["
      "\U0001F600-\U0001F64F"  # emoticons
      "\U0001F300-\U0001F5FF"  # symbols & pictographs
      "\U0001F680-\U0001F6FF"  # transport & map symbols
      "\U0001F1E0-\U0001F1FF"  # flags (iOS)
      "\U00002702-\U000027B0"
      "\U000024C2-\U0001F251"
      "]+",
      flags=re.UNICODE
  )
  return emoji_pattern.sub(r'', text)
def preprocess_text(text):
  text = text.lower()
  text = remove_punctuation(text)
  text = remove_emoji(text)
  text = ' '.join(text.split())
  return text


def preprocess_dataset(ds, lang_src, lang_tgt):
    def preprocess_fn(batch):
        return {
            lang_src: preprocess_text(batch[lang_src]),
            lang_tgt: preprocess_text(batch[lang_tgt])
        }

    ds = ds.map(preprocess_fn)
    return ds



def load_parallel_txt(src_path: str,
                      tgt_path: str,
                      src_lang: str,
                      tgt_lang: str,
                      encoding: str = "utf-8",
                      skip_empty_pairs: bool = True):
  """
  Baca dua file teks paralel dan kembalikan datasets.Dataset dengan kolom:
  - src_lang (mis. 'indonesian')
  - tgt_lang (mis. 'english')

  Jika jumlah baris berbeda, dataset akan dipotong ke panjang minimal.
  Jika skip_empty_pairs=True, baris di mana kedua sisi kosong akan di-skip.
  """
  # baca semua baris, pertahankan urutan; hapus newline/carriage returns
  with io.open(src_path, "r", encoding=encoding) as f:
    src_lines = [line.rstrip("\r\n") for line in f]
  with io.open(tgt_path, "r", encoding=encoding) as f:
    tgt_lines = [line.rstrip("\r\n") for line in f]
  print("open file selesai")

  # jika panjang berbeda, potong ke panjang minimum dan beri peringatan
  if len(src_lines) != len(tgt_lines):
    min_len = min(len(src_lines), len(tgt_lines))
    print(f"[warning] jumlah baris berbeda: src={len(src_lines)} tgt={len(tgt_lines)}. Memotong ke {min_len} pasangan.")
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]
    print("panjang src dan tgt tidak sama")

  # buat list pasangan dict sesuai nama kolom (sesuaikan dengan config)
  pairs = []
  for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
    s_stripped = s.strip()
    t_stripped = t.strip()
    if skip_empty_pairs and s_stripped == "" and t_stripped == "":
      # lewati pasangan kosong kedua sisi
      continue
    pairs.append({src_lang: s_stripped, tgt_lang: t_stripped, "id": i})
  print(f"total pasangan setelah skip empty: {len(pairs)}")
  print(pairs[:2])
  # buat datasets.Dataset
  ds = Dataset.from_list(pairs)
  print("membuat dataset selesai")
  return ds


# --- penggunaan (menggantikan bagian load_dataset Anda) ---
# diasumsikan config sudah ada dan berisi config["lang_src"] dan config["lang_tgt"],
# mis. config["lang_src"] = "indonesian", config["lang_tgt"] = "english"

# ds_raw = load_parallel_txt("indonesian.txt", "english.txt",
#                            src_lang=config["lang_src"],
#                            tgt_lang=config["lang_tgt"])

# # lalu lanjutkan seperti semula
# ds_raw = preprocess_dataset(ds_raw, config["lang_src"], config["lang_tgt"])
# tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
# tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

def get_ds(config):
  # ds_raw = load_dataset("jakartaresearch/inglish", split='train')
  # ds_raw = load_dataset("jakartaresearch/inglish", split='train')
  # dataset = load_dataset("csv", data_files="path/ke/file.csv", revision="nama_branch")
  ds_raw = load_dataset(
    "jakartaresearch/inglish",
    data_files="default/train/0000.parquet",
    split="train",
    revision="refs/convert/parquet"
  )
  # ds_raw = load_parallel_txt("en-id.txt-ted/TED2020.en-id.id", "en-id.txt-ted/TED2020.en-id.en", src_lang=config["lang_src"], tgt_lang=config["lang_tgt"])

  # lalu lanjutkan seperti semula
  ds_raw = preprocess_dataset(ds_raw, config["lang_src"], config["lang_tgt"])
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  ds_raw = preprocess_dataset(ds_raw, config["lang_src"], config["lang_tgt"])
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  train_ds_size = int(len(ds_raw) * 0.9)
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item[config["lang_src"]]).ids
    tgt_ids = tokenizer_tgt.encode(item[config["lang_tgt"]]).ids

    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print("max length of source sentence", max_len_src)
  print("max length of target sentence", max_len_tgt)

  train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len, N, h):
  model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], N=N, h=h)
  return model

def train_model(config, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device : ", device)

  Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
  # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=config)
  model = get_model(config=config, vocab_src_len=tokenizer_src.get_vocab_size(), vocab_tgt_len=tokenizer_tgt.get_vocab_size(), N=config["N"], h=config["h"]).to(device=device)

  # tensorboard
  writer = SummaryWriter(config["experiment_name"])

  optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

  initial_epoch = 0
  global_step = 0
  if config["preload"]:
    model_filename = get_weights_file_path(config, config["preload"])
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    initial_epoch = state["epoch"] + 1
    optimizer.load_state_dict(state["optimizer_state_dict"])
    global_step = state["global_step"]

  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device=device)
  for epoch in range(initial_epoch, config["num_epochs"]):
    batch_iterator = tqdm(train_dataloader, desc=f"Processing  epoch {epoch:02d}")
    for batch in batch_iterator:
      model.train()
      encoder_input = batch["encoder_input"].to(device)
      decoder_input = batch["decoder_input"].to(device)
      encoder_mask = batch["encoder_mask"].to(device)
      decoder_mask = batch["decoder_mask"].to(device)

      encoder_output = model.encode(encoder_input, encoder_mask)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # type: ignore
      proj_output = model.projection_layer(decoder_output)

      label = batch["label"].to(device)

      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

      batch_iterator.set_postfix({"Loss" : f"{loss.item():6.3f}"})

      # log loss
      writer.add_scalar("train loss", loss.item(), global_step)
      writer.flush()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)

      global_step += 1

    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save(
      {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict" : optimizer.state_dict(),
      "global_step": global_step
    },
    model_filename
    )

    # if (epoch + 1) % 1 == 0 or (epoch + 1) == config["num_epochs"]:

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
  sos_idx = tokenizer_tgt.token_to_id("[SOS]")
  eos_idx = tokenizer_tgt.token_to_id("[EOS]")

  encoder_output = model.encode(source, source_mask)

  decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
  while True:
    if decoder_input.size(1) == max_len:
      break
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    prob = model.projection_layer(out[:, -1])

    _, next_word = torch.max(prob, dim = 1)
    decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
    if next_word == eos_idx:
      break
  return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2, return_text=False):
  model.eval()
  count = 0

  # source_texts = []
  # references = []
  # predictions = []

  console_width = 80

  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch["encoder_input"].to(device)
      encoder_mask = batch["encoder_mask"].to(device)
      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

      source_text = batch["src_text"][0]
      target_text = batch["tgt_text"][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      # source_texts.append(source_text)
      # references.append(target_text)
      # predictions.append(model_out_text)

      print_msg("-" * console_width)
      print_msg(f"Source: {source_text}")
      print_msg(f"Target: {target_text}")
      print_msg(f"predictions: {model_out_text}")

      if count == num_examples:
        break
  
  # if writer:
  #   writer.flush()
  #   # compute metrics like BLEU, WER, CER, etc here
  #   # bleu_metric = load("bleu")
  #   # print_msg(f"BLEU: {bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])['bleu']*100:.2f}")
  #   bleu_metric = load("bleu")
  #   print_msg(f"BLEU: {bleu_metric.compute(predictions=[predictions], references=[[target_text]])['bleu']*100:.2f}")
    # torchmetrics CharErrorRate, BLUE, WordErrorRate, etc can be used here
    # pass
  # if return_text:
  #   return predictions

def majority_vote(models, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
  # model.eval()
  count = 0

  # source_texts = []
  references = []
  predictions = {}
  predictions["majority_vote"] = []
  for model_name in models.keys():
    references[model_name] = []
    predictions[model_name] = []
  # bleu_metric = load("bleu")

  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch["encoder_input"].to(device)
      encoder_mask = batch["encoder_mask"].to(device)
      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
      # source_text = batch["src_text"][0]
      target_text = batch["tgt_text"][0]
      references.append(target_text)
      for model_name, model in models.items():
        model.eval()
        model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        # references[model_name].append(target_text)
        predictions[model_name].append(model_out_text)

  for idx in list(models.keys())[-1]:
    list_of_predictions = []
    max_len = 0
    for model_name in models.keys():
      pred = predictions[model_name][idx].split()
      list_of_predictions.append(pred)
      max_len = max(max_len, len(pred))
    for i in range(max_len):
      word_count = {}
      for pred in list_of_predictions:
        if i < len(pred):
          word = pred[i]
          if word in word_count:
            word_count[word] += 1
          else:
            word_count[word] = 1
      if word_count:
        majority_word = max(word_count, key=word_count.get)
        predictions["majority_vote"].append(majority_word)
      else:
        break
  return predictions, references
    # predictions["majority_vote"] = []



      # source_texts.append(source_text)

      # print_msg("-" * console_width)
      # print_msg(f"Source: {source_text}")
      # print_msg(f"Target: {target_text}")
      # print_msg(f"predictions: {model_out_text}")

      # if count == num_examples:
      #   return source_texts, references, predictions
        # break
    # print_msg(f"BLEU: {bleu_metric.compute(predictions=[model_out_text], references=[[target_text]])['bleu']*100:.2f}")