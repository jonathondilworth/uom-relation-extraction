# UoM Relation Extraction Project

Relation Extraction for Text Mining - COMP61332

## Statement of authorship

The codebase is from the paper ["An Improved Baseline for Relation Extraction"](https://github.com/wzhouad/RE_improved_baseline).

## Summary of changes

The following files were modified from the original code base (comments in files show where edits have been made):
- prepro.py: added 7 novel approaches to marked entities 
- train_retacred.py
- train_tacred.py
- utils.py: added the ability to save models

The following file were added:
- inference.py : this allows you to get the dev and test f1 scores for a particular model
- inference_sent.py: this allows a user to write a sentence in command line and receive the model's prediction



## Project Structure

*Note: due to licensing, the dataset is not included within this repository. Instructions on where to obtain the dataset and how to patch it can be found under the next section.*

```bash
.root_dir
│
├── config/
│   └── ...
├── data/
│   ├── retacred/
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   └── tacred/
│       ├── dev.json
│       ├── test.json
│       └── train.json
├── docs/
│   └── ...
├── notebooks/
│   └── experimental/
│       ├── utils/
│       └── models/
│           └── ...
├── paper/
│   ├── draft.pdf
│   ├── final.pdf
│   └── TeX/
│       └── ...
├── scripts/
│   └── ...
├── src/
│   └── utils/
│       └── ...
├── tests/
│   └── ...
├── .gitignore
├── environment.yml
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

*We may consider merging experimental notebooks into 00_experimental.ipynb, retaining that folder on a seperate branch and eventually remove it from 'main'.*


