import os
import torch
from transformers import AutoConfig, AutoTokenizer
from prepro import RETACREDProcessor
from model import REModel
from train_retacred import evaluate
from utils import set_seed

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


def main():
  # mocking arguments temporarily
  args = arguments()
  # reminder: don't pass args.seed to set_seed, it breaks.
  set_seed(args)
  # auto-load the config & tokenizer by providing the path to the checkpoint/best saved model directory
  config = AutoConfig.from_pretrained(args.load_path)
  tokenizer = AutoTokenizer.from_pretrained(args.load_path)
  print(f"Loading weights from {args.load_path} ...")
  # see: gcn-over-pruned-trees/utils/torch_utils.py - line 144 (def load(...))
  checkpoint = torch.load(args.model_name_or_path, map_location=args.device)
  
  # model is wrapping torch.nn.Module, sets the params/config & handles the forward-pass
  model = REModel(args, config)
  # note: if you're loading best_model.pt <- may have to swap out 'model_state_dict' for 'model', depending on how 'params' to save was specified
  # for checkpoints, specify: 'model_state_dict' and for 'best_model.pt' use 'model' <-- TODO: fix this
  model.load_state_dict(checkpoint['model'])
  model.to(device=args.device)
  # evaluate from train_retacred runs model.eval()
  # model.eval()

  # always gives me this warning: 'You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.'
  # I'm not too sure why... it still evaluates to the fine-tuned dev F1 & test F1 just fine...
  
  # Warning, if you load a checkpoint and have since added any new tokens to the vocab, you'll see the following warning:
  # "Some weights of BertModel were not initialized from the model checkpoint at saved_models_ber_base/best_model.pt and are newly initialized"
  
  # we need the TACRED/RETACRED Processor for tokenisation; and for injecting special characters into the input encoding, see: 'input_format' 
  processor = RETACREDProcessor(args, tokenizer)
  dev_file = os.path.join(args.data_dir, "dev.json")
  test_file = os.path.join(args.data_dir, "test.json")
  dev_features = processor.read(dev_file)
  test_features = processor.read(test_file)

  # Evaluate on dev
  print("\n")
  print("Evaluating dev set ...")
  dev_f1, dev_out = evaluate(args, model, dev_features, tag='dev')

  # Evaluate on test
  print("\n")
  print("Evaluating test set ...")
  test_f1, test_out = evaluate(args, model, test_features, tag='test')

  # Print final results
  print("\n")
  print("Results:")
  print("Dev F1: ", dev_f1 * 100.0)
  print("Test F1: ", test_f1 * 100.0)

if __name__ == "__main__":
  main()