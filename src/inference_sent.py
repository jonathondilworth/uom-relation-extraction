import os
import torch
import argparse
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from prepro import RETACREDProcessor
from model import REModel
from utils import set_seed, collate_fn
import torch.nn.functional as F


class ModelArguments:
  def __init__(self, args=None):
    if args:
      for k, v in vars(args).items():
        setattr(self, k, v)
    else:
      self.data_dir = "./data/retacred"
      self.model_name_or_path = "saved_models_ber_base/best_model.pt"
      self.input_format = "typed_entity_marker_punct"
      self.max_seq_length = 512
      self.test_batch_size = 32
      self.num_classes = 40
      self.dropout_prob = 0.1
      self.load_path = "saved_models_ber_base" 
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.n_gpu = torch.cuda.device_count()
      self.seed = 42
      self.num_class = 40
class UserInput:
  def __init__(self, args):
    self.sentence = args.sentence
    self.tokens = args.tokens.split(",")
    self.pos = args.pos.split(",")
    self.deprel = args.deprel.split(",")
    self.subj_start = args.subj_start
    self.subj_end = args.subj_end
    self.obj_start = args.obj_start
    self.obj_end = args.obj_end
    self.subj_type = args.subj_type
    self.obj_type = args.obj_type


def get_user_input():
  parser = argparse.ArgumentParser(description="Run inference for relation extraction")
  parser.add_argument("--sentence", type=str, required=True, help="Input sentence")
  parser.add_argument("--tokens", type=str, required=True, help="Comma-separated list of tokens")
  parser.add_argument("--pos", type=str, required=True, help="Comma-separated list of POS tags")
  parser.add_argument("--deprel", type=str, required=True, help="Comma-separated list of dependency relations")
  parser.add_argument("--subj_start", type=int, required=True, help="Start index of subject entity")
  parser.add_argument("--subj_end", type=int, required=True, help="End index of subject entity")
  parser.add_argument("--obj_start", type=int, required=True, help="Start index of object entity")
  parser.add_argument("--obj_end", type=int, required=True, help="End index of object entity")
  parser.add_argument("--subj_type", type=str, required=True, help="Type of subject entity")
  parser.add_argument("--obj_type", type=str, required=True, help="Type of object entity")
  return parser.parse_args()


# see: https://stackoverflow.com/questions/45384684/replace-all-nonzero-values-by-zero-and-all-zero-values-by-a-specific-value
def custom_replace(tensor, on_zero, on_non_zero, cast_to=torch.float):
    res = tensor.clone().to(cast_to)
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res

# why do they have to use the most annoying conventions
# e.g. in utils.py, they use 'f' to index into each batches input ids:
# [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
# firstly, f is used to format strings, so that's confusing
# secondly, descriptive variable names, please!

# also, tensors, use them!
# https://stackoverflow.com/questions/48686945/reshaping-a-tensor-with-padding-in-pytorch

# adapted version of collate_fn from utils.py (as we're using a batch_size of 1)
def collate_single(input_ids, ss, os, max_seq_length=512):
  # we've got problems if the max sequence length is < input length
  if len(input_ids) > max_seq_length:
   raise Exception("Your input is larger than your maximum sequence length.")
  # padding up to the max_seq_length with zeros
  tensor_ids = torch.tensor(input_ids)
  input_ids_tensor = F.pad(
    tensor_ids, 
    pad=(0, max_seq_length - tensor_ids.shape[0]),
    mode='constant', 
    value=0.0
  ).unsqueeze(0)
  # replicating the attention mask in utils.py (collate_fn)
  attention_tensor = custom_replace(input_ids_tensor, 0.0, 1.0)
  ss_tensor = torch.tensor([ss], dtype=torch.long);
  os_tensor = torch.tensor([os], dtype=torch.long);
  return input_ids_tensor, attention_tensor, ss_tensor, os_tensor

def main():
  user_args = get_user_input()
  user_input = UserInput(user_args)

  args = ModelArguments()
  set_seed(args)

  # implemented currently for retacred - TODO: update this so its 'modifiable'
  LABEL_TO_ID = {0: 'no_relation', 1: 'org:founded_by', 2: 'per:identity', 3: 'org:alternate_names', 4: 'per:children', 5: 'per:origin', 6: 'per:countries_of_residence', 7: 'per:employee_of', 8: 'per:title', 9: 'org:city_of_branch', 10: 'per:religion', 11: 'per:age', 12: 'per:date_of_death', 13: 'org:website', 14: 'per:stateorprovinces_of_residence', 15: 'org:top_members/employees', 16: 'org:number_of_employees/members', 17: 'org:members', 18: 'org:country_of_branch', 19: 'per:spouse', 20: 'org:stateorprovince_of_branch', 21: 'org:political/religious_affiliation', 22: 'org:member_of', 23: 'per:siblings', 24: 'per:stateorprovince_of_birth', 25: 'org:dissolved', 26: 'per:other_family', 27: 'org:shareholders', 28: 'per:parents', 29: 'per:charges', 30: 'per:schools_attended', 31: 'per:cause_of_death', 32: 'per:city_of_death', 33: 'per:stateorprovince_of_death', 34: 'org:founded', 35: 'per:country_of_death', 36: 'per:country_of_birth', 37: 'per:date_of_birth', 38: 'per:cities_of_residence', 39: 'per:city_of_birth'}

  # load config, a lot of the same as inference.py
  config = AutoConfig.from_pretrained(args.load_path)
  tokenizer = AutoTokenizer.from_pretrained(args.load_path)
  
  processor = RETACREDProcessor(args, tokenizer)
  prepd_input_representation = processor.tokenize(user_input.tokens, user_input.subj_type, user_input.obj_type, user_input.subj_start, user_input.subj_end, user_input.obj_start, user_input.obj_end, user_input.pos, user_input.deprel)


  input_ids_tensor, attention_tensor, ss_tensor, os_tensor = collate_single(
    prepd_input_representation[0],
    prepd_input_representation[1],
    prepd_input_representation[2]
  )

     
  # more of the same from inference.py
  print(f"Loading weights from {args.load_path} ...")
  checkpoint = torch.load(args.model_name_or_path, map_location=args.device)

  model = REModel(args, config)
  #TODO: vocab size mismatch issue -- is this just me?
  old_vocab_size = checkpoint['model_state_dict']['encoder.embeddings.word_embeddings.weight'].shape[0] 
  new_vocab_size = model.encoder.embeddings.word_embeddings.weight.shape[0]

  if old_vocab_size != new_vocab_size:
    print(f"Resizing model embeddings from {new_vocab_size} to {old_vocab_size} to match checkpoint")
    model.encoder.resize_token_embeddings(old_vocab_size)

  #TODO-- orginally checkpoint['model'], mine is checkpoint['model_state_dict']??
  model.load_state_dict(checkpoint['model_state_dict']) 
  
  model.to(device=args.device)
  model.eval()

  # print(args.device)

  # modified from evaluate in train_retacred & train_tacred
  with torch.no_grad():
    inputs = {
        'input_ids': input_ids_tensor.to(args.device),
        'attention_mask': attention_tensor.to(args.device),
        'ss': ss_tensor.to(args.device),
        'os': os_tensor.to(args.device)
    }
    logits = model(**inputs)[0]
    pred = torch.argmax(logits, dim=-1).item()

  
  # the correct label is "no relation"
  # TODO: try some more relations!
  relation_label = LABEL_TO_ID.get(pred, "unknown relation")

  print("\n")
  print("-----")
  print(f"Sentence Tokens: {user_input.sentence}")
  print(f"Predicted Relation: {relation_label} ({pred})")
  print(f"Logits: {logits.cpu().numpy()}\n")
  print("-----")

if __name__ == "__main__":
  main()