# UoM Relation Extraction Project

Relation Extraction for Text Mining - COMP61332

**Note: this README.md is a Work in Progress!**

## Project Structure

*Note: due to licensing, the dataset is not included within this repository. Instructions on where to obtain the dataset and how to patch it can be found under the next section.*

'''
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
'''

*We may consider merging experimental notebooks into 00_experimental.ipynb, retaining that folder on a seperate branch and eventually remove it from 'main'.*


