import os
import torch
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from prepro import RETACREDProcessor
from model import REModel
from utils import set_seed, collate_fn
import torch.nn.functional as F

# I'm just spoofing/mocking the arguments for the time being
class arguments:
  def __init__(self, args=None):
    if args:
      for k, v in args:
        self.k = v
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

  args = arguments()
  set_seed(args)

  # implemented currently for retacred - TODO: update this so its 'modifiable'
  LABEL_TO_ID = {0: 'no_relation', 1: 'org:founded_by', 2: 'per:identity', 3: 'org:alternate_names', 4: 'per:children', 5: 'per:origin', 6: 'per:countries_of_residence', 7: 'per:employee_of', 8: 'per:title', 9: 'org:city_of_branch', 10: 'per:religion', 11: 'per:age', 12: 'per:date_of_death', 13: 'org:website', 14: 'per:stateorprovinces_of_residence', 15: 'org:top_members/employees', 16: 'org:number_of_employees/members', 17: 'org:members', 18: 'org:country_of_branch', 19: 'per:spouse', 20: 'org:stateorprovince_of_branch', 21: 'org:political/religious_affiliation', 22: 'org:member_of', 23: 'per:siblings', 24: 'per:stateorprovince_of_birth', 25: 'org:dissolved', 26: 'per:other_family', 27: 'org:shareholders', 28: 'per:parents', 29: 'per:charges', 30: 'per:schools_attended', 31: 'per:cause_of_death', 32: 'per:city_of_death', 33: 'per:stateorprovince_of_death', 34: 'org:founded', 35: 'per:country_of_death', 36: 'per:country_of_birth', 37: 'per:date_of_birth', 38: 'per:cities_of_residence', 39: 'per:city_of_birth'}

  # load config, a lot of the same as inference.py
  config = AutoConfig.from_pretrained(args.load_path)
  tokenizer = AutoTokenizer.from_pretrained(args.load_path)

  # input_ids, labels, ss, os
  example_sentance = "He has served as a policy aide to the late U.S. Senator Alan Cranston , as National Issues Director for the 2004 presidential campaign of Congressman Dennis Kucinich , as a co-founder of Progressive Democrats of America and as a member of the international policy department at the RAND Corporation think tank before all that ."
  example_tokens = [ "He", "has", "served", "as", "a", "policy", "aide", "to", "the", "late", "U.S.",
                    "Senator", "Alan", "Cranston", ",", "as", "National", "Issues", "Director", "for", "the", "2004", 
                    "presidential", "campaign", "of", "Congressman", "Dennis", "Kucinich", ",", "as", "a", "co-founder", "of", 
                    "Progressive", "Democrats", "of", "America", "and", "as", "a", "member", "of", "the", "international", 
                    "policy", "department", "at", "the", "RAND", "Corporation", "think", "tank", "before", "all", "that", "." ]
  example_pos = [ "PRP", "VBZ", "VBN", "IN", "DT", "NN", "NN", "TO", "DT", "JJ", "NNP", 
                  "NNP", "NNP", "NNP", ",", "IN", "NNP", "NNP", "NNP", "IN", "DT", "CD",
                  "JJ", "NN", "IN", "NNP", "NNP", "NNP", ",", "IN", "DT", "NN", "IN", 
                  "NNP", "NNPS", "IN", "NNP", "CC", "IN", "DT", "NN", "IN", "DT", "JJ", 
                  "NN", "NN", "IN", "DT", "NNP", "NNP", "VB", "NN", "IN", "DT", "DT", "." ]
  example_deprel = [ "nsubj", "aux", "ROOT", "case", "det", "compound", "nmod", "case", "det", "amod",
                     "compound","compound","compound","nmod","punct","case","compound","compound","nmod","case","det", 
                     "nummod", "amod", "nmod", "case", "compound", "compound", "nmod", "punct", "case", "det", "nmod", 
                     "case", "compound", "nmod", "case", "nmod", "cc", "case", "det", "conj", "case", "det", 
                     "amod", "compound", "nmod", "case", "det", "compound", "nmod", "acl", "dobj", "case", "nmod", "dep", "punct" ]
  example_subj_start = 33
  example_subj_end = 36
  example_obj_start = 43
  example_obj_end = 45
  example_subj_type ="ORGANIZATION"
  example_obj_type = "ORGANIZATION"

  # sanity check:
  print(f"example sentance: {example_sentance}")
  print(f"example tokens length: {len(example_tokens)}")
  print(f"example pos length: {len(example_pos)}")
  print(f"example deprel length: {len(example_deprel)}")

  print(f"example subj start: {example_subj_start}")
  print(f"example subject end: {example_subj_end}")
  
  # subject sanity check:
  subj_length = example_subj_end - example_subj_start
  for x in range(subj_length + 1):
    print(example_tokens[example_subj_start + x])

  # object sanity check:
  obj_length = example_obj_end - example_obj_start
  for x in range(obj_length + 1):
    print(example_tokens[example_obj_start + x])

  # print("\n\n")

  processor = RETACREDProcessor(args, tokenizer)
  prepd_input_representation = processor.tokenize(example_tokens, example_subj_type, example_obj_type, example_subj_start, example_subj_end, example_obj_start, example_obj_end, example_pos, example_deprel)

  # sanity check:
  print(prepd_input_representation)

  # double sanity check:
  print("input_ids: ", prepd_input_representation[0])
  print("ss: ", prepd_input_representation[1])
  print("os: ", prepd_input_representation[2])

  # batch of size 1:
  # single_batch_input = collate_fn({
  #   "input_ids": prepd_input_representation[0],
  #   "labels": 0,
  #   "ss": prepd_input_representation[1],
  #   "os": prepd_input_representation[2]
  # })

  input_ids_tensor, attention_tensor, ss_tensor, os_tensor = collate_single(
    prepd_input_representation[0],
    prepd_input_representation[1],
    prepd_input_representation[2]
  )

  # sanity check
  print(input_ids_tensor)
  print(attention_tensor)
  print(ss_tensor)
  print(os_tensor)
     
  # more of the same from inference.py
  print(f"Loading weights from {args.load_path} ...")
  checkpoint = torch.load(args.model_name_or_path, map_location=args.device)

  model = REModel(args, config)
  model.load_state_dict(checkpoint['model'])
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
  print(f"Sentence Tokens: {example_sentance}")
  print(f"Predicted Relation: {relation_label} ({pred})")
  print(f"Logits: {logits.cpu().numpy()}\n")
  print("-----")

if __name__ == "__main__":
  main()