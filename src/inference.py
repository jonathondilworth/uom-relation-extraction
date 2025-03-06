import os
import torch
import argparse
from transformers import AutoConfig, AutoTokenizer
from prepro import RETACREDProcessor
from model import REModel
from train_retacred import evaluate
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a relation extraction model using RETACRED data."
    )
    parser.add_argument("--data_dir", type=str, default="./data/retacred", help="Path to the data directory.")
    parser.add_argument("--model_checkpoint", type=str, default="saved_models_ber_base/best_model.pt", help="Path to the model checkpoint file.")
    parser.add_argument("--load_path", type=str, default="saved_models_ber_base", help="Directory path for saved model (config & tokenizer).")
    parser.add_argument("--input_format", type=str, default="typed_entity_marker_punct", help="Input format for tokenization.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for inputs.")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_classes", type=int, default=40, help="Number of relation classes.")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def main():
  args = parse_args()
  args.model_name_or_path = args.load_path
  
  #device attribute
  args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args.n_gpu = torch.cuda.device_count() 
  args.num_class = args.num_classes 

  # reminder: don't pass args.seed to set_seed, it breaks.
  set_seed(args)
  # auto-load the config & tokenizer by providing the path to the checkpoint/best saved model directory
  config = AutoConfig.from_pretrained(args.load_path)
  tokenizer = AutoTokenizer.from_pretrained(args.load_path)
  print(f"Loading weights from {args.load_path} ...")
  # see: gcn-over-pruned-trees/utils/torch_utils.py - line 144 (def load(...))
  checkpoint = torch.load(args.model_checkpoint, map_location=args.device)
  
  # model is wrapping torch.nn.Module, sets the params/config & handles the forward-pass
  model = REModel(args, config)
  model.encoder.resize_token_embeddings(len(tokenizer))
  # note: if you're loading best_model.pt <- may have to swap out 'model_state_dict' for 'model', depending on how 'params' to save was specified
  # for checkpoints, specify: 'model_state_dict' and for 'best_model.pt' use 'model' <-- TODO: fix this
  # model.load_state_dict(checkpoint['model'])
  if 'model_state_dict' in checkpoint:
      model.load_state_dict(checkpoint['model_state_dict'])
  else:
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