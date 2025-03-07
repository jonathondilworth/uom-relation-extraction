import torch
import random
import numpy as np
import os

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0 and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    output = (input_ids, input_mask, labels, ss, os)
    return output

#######
# CHANGES: model checkpointing
# gcn-over-pruned-trees comes bundled with a nicer set of patterns (model checkpointing being one of them)
# see: gcn-over-pruned-trees/utils/torch_utils.py
#######

# TODO: think about implementing a means of saving to a new directory when repeating experiments in a bash loop
# (or just use the loop iteration as: {save_dir}_$I/checkpoint-{step} .. )

# we're passing the args in rather than the filename explicitly, then constructing the filepath
def save_checkpoint(args, model, optimizer, scheduler, step):
    # for now, just save to an existing directory if one exists for the specified save_dir (or create one)
    if not os.path.exists(args.save_dir):
        print(f"creating new checkpoint directory at: {args.save_dir}")
        os.makedirs(args.save_dir)

    # filepath/filename
    checkpoint_fp = os.path.join(args.save_dir, f"checkpoint-{step}.pt")

    # in case we want to load to resume training, includes step, etc
    params = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    # ^^
    # we could save the config in the dump, but it would be kind of redundant, since we're dumping the config via AutoConfig
    # see: line 221, 222 (train_tacred.py) & 228, 229 (train_retacred.py)
    # might be a better design pattern to include it in the dump, not too sure about this though

    print(f"[ Trying to save checkpoint to: {checkpoint_fp} ] ... ")

    # 'save' design pattern as outlined in:
    # https://github.com/qipeng/gcn-over-pruned-trees/blob/master/utils/torch_utils.py
    # ^ line 133

    try:
      torch.save(params, checkpoint_fp)
      print(f"model saved to: {checkpoint_fp}")
    except Exception:
       print("[ Warning: saving failed... continuing anyway. ... ]")


def save_best(args, model):
    # ensure dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    best_model_dir = os.path.join(args.save_dir, f"pytorch_model.bin")

    # <s>presumably, if we're saving the 'best' model, we're not neccesarily planning on continuing to train it</s>
    # ^ maintaining consistent naming convention, model -> model_state_dict
    params = {
        'model_state_dict': model.state_dict()
    }

    try:
        torch.save(params, best_model_dir)
    except Exception:
        print("[ Warning: saving failed... continuing anyway. ... ]")

#######
# END CHANGES
#######
