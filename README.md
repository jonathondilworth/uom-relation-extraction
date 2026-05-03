# Relation Extraction for Text Mining

Relation extraction (RE) is a fundamental task in text mining. While many modern approaches to RE utilise neural architectures, these architectures can differ substantially. For instance, the use of BERT-based language models and graph convolutional neural networks (GCNs) are both well-known techniques for effective RE. However, incorporating attention over the output embeddings of a GCN and exploring complex part-of-speech (PoS) tag enrichment during BERT-based fine-tuning remains underexplored. This work aims to bridge that gap by utilising scaled dot-product attention over the outputs of a GCN, varying the PoS tag representation, and enriching dependency relations with varying BERT-based entity markers during fine-tuning. We (1) compare our approaches to conventional implementations, and (2) compare the two approaches with one another in an exploratory manner. We find that our variations perform slightly worse than conventional implementations, but can improve computational efficiency during training for the GCN model. In addition, the BERT model outperforms the GCN-based approach in all cases. Future work could further improve the training efficiency of our GCN-based approach and investigate whether BERT-based enhancements may be beneficial for domain-specific RE (e.g., biomedical, legal, or scientific text), where explicit syntactic cues may be more useful.

**View the paper:** [https://dilworth.me/PoSTACRED.pdf](https://dilworth.me/PoSTACRED.pdf)

**Authors (equal contribution):** Jonathon Dilworth, Emma O’Brien, Rojs Aktumanis, Alexandros Michaelides

The TeX is available within the [paper directory](/paper).

_Note: this project was a collaborative research effort undertaken during the pursuit of all authors' MSc degrees and is largely exploratory._

## Experiments

The experimental code is provided in the [notebooks](/notebooks) for reproducibility.

## Data

Due to licensing restrictions, the dataset is not included in this repository.

Instructions on where to obtain the dataset and how to patch it can be found within the project [documentation](/docs).

## Project Structure

```
.root_dir
│
├── config/
│   └── ...
├── data/
│   ├── retacred/
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── train.json
│   └── tacred/
│       ├── dev.json
│       ├── test.json
│       └── train.json
├── docs/
│   └── ...
├── notebooks/
│   └── experimental/
│       ├── utils/
│       └── models/
│           └── ...
├── paper/
│   ├── draft.pdf
│   ├── final.pdf
│   └── TeX/
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

## Statement of authorship

Part of this codebase is inherited from the paper “An Improved Baseline for Relation Extraction”.

## License

MIT License

