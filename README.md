# Zeal: Zero-Effort Automatic Learning

Zeal is a machine learning tool to build NLP models for ranking, classification and sequence tagging problems. The goal of Zeal is to train NLP model with zero effort but still achieve decent performance. It also has model insight component for comparing parameters and metrics of trained models.

## How to use

1. Install python packages specified in requirement.txt.
2. Choose problem type and prepare training data in the format defined in `example_conf/README.md`.
3. Run `python main.py train --conf example_conf/ranking.conf --note "some notes"`. The best model is automatically selected and saved to `./output`. Model folder is named by running date.
4. Run `python logdb.py` to generate model comparison result in `model.csv`. 
5. If you want to serve trained model with tensorflow serving, run `python main.py export --model_path output/MODEL_FOLDER` to export it to servable model, which will be saved under `output/MODEL_FOLDER/serve`.