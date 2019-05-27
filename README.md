# Zeal: Zero-Effort Agile Learning

Zeal is a machine learning tool to build NLP models for ranking, classification and sequence tagging problems. The goal of Zeal is to train NLP model with zero effort (user still need to prepare training and evaluation data in the given format), and achieve decent performance. Zeal also manages models and training results in a database, and user can easily compare parameters and metrics of previous trained models.

## How to use

1. Install python packages specified in requirement.txt.
2. Choose problem type and prepare training data in the format defined in example_conf/README.md
3. Run `python main.py train --conf example_conf/ranking.conf`. The best model is automatically selected and saved to `./output`.
4. Run `python logdb.py` to generate model comparison result in `model.csv`. 