# UoM Relation Extraction Project

Relation Extraction for Text Mining - COMP61332

## Statement of authorship

The codebase from the paper ["An Improved Baseline for Relation Extraction"](https://github.com/wzhouad/RE_improved_baseline).

The following files were modified from the original code base:
- prepro.py
- train_retacred.py
- train_tacred.py
- utils.py

The following file were added:
- inference.py
- inference_sent.py
- invert_label_2_id.py


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


