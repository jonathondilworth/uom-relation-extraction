
## Statement of authorship

The codebase presented is mostly derived from two other groups of authors. Firstly, we used the original code for C-GCN and GCN (https://github.com/qipeng/gcn-over-pruned-trees/tree/master). Secondly, the adaptations to make it compatible with Re-TACRED were defined by the authors of Re-TACRED (https://github.com/gstoica27/Re-TACRED). In this section I present the codebase changes that were made to enable our implementation of the GCN model.  
The following files were modified:
- train.py
- data/loader.py
- model/trainer.py
- model/gcn.py (MOST IMPORTANT)  

## Summary of changes

### train.py
Added support for parsing new input arguments specific to the new model

### data/loader.py
Adapted the existing data loader to allow inference, not just training/testing

### model/trainer.py
Adapted the existing data loader to allow inference, not just training/testing

### model/gcn.py
- Adapted the C-GCN model to run without presence of CUDA.  
- Changed the model. Can run the model with or without using the sentence embeddings as input to the classifier. Can add positional embeddings to the output of GCN. Can use multi-head attention either before pooling or after pooling.

