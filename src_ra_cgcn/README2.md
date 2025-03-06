
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
Adapted the existing trainer to allow inference, not just training/testing

### model/gcn.py
- Adapted the C-GCN model to run without presence of CUDA.  
- Changed the model. Can run the model with or without using the sentence embeddings as input to the classifier. Can add positional embeddings to the output of GCN. Can use multi-head attention either before pooling or after pooling.


## Notebooks
- Training notebook. The training notebook showcases how training and hyperparameter tuning was done. We proposed multiple architectural changes to the GCN model, and found the best ones by running the scripts in the hyperparameter tuning part. Please be aware that the training might take a long time (1 min/epoch on 16Gb M2 MacBook for each model), and hyperparameter tuning runs 32 of these models for 100 epochs.

- Testing notebook. The testing notebook showcases how a model can be loaded, and how testing and inference can be done. It gives examples of inference and includes guidlines on how to format the input.
