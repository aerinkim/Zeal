# Configurations
Some configuration file examples are provided in this folder. The configurable parameters are listed below:

## problem_type
The problem type in training phase. Currently support:

1. `pointwise_ranking`: pointwise ranking via binary classification.
2. `classification`: multi-class classifcation.
3. `sequence_tag`: sequence tagging.

## sep (ranking and sequence tagging problem only)
Separator to use when splitting the targets in ranking problem, and query and tagging labels in sequence tagging problem. Default is `|`.

## neg\_num (ranking problem only)
Number of negative targets for each query in training data. Default is `-1`, which is to take all targets except for positive ones as negative targets. 

## bert\_base
The bert base model to use. Set `base` to use BERT-Base, or `large` to use BERT-Large. 

## do\_lower\_case
Whether to ignore case (`True`) or not (`False`).

## train\_data\_path
Training data path.

## dev\_data\_path
Dev data path. Model will be selected based on dev data metrics.

## max\_seq\_length
Max sequence length of BERT input.

## n\_fine\_tune\_layers
Number of BERT layers to fine-tune. Set `0` to fix all BERT layers, and `-1` to fine-tune all BERT layers.

## learning\_rate
Learning rate for Adam optimizer.

## epochs
Training epochs.

## batch\_size
Training batch size.