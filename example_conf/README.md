#Configurations

##problem_type
The problem type in training phase. Currently support:

**pointwise_ranking**: pointwise ranking via binary classification

Training data format:

query[tab]pos_target1|pos_target1|...

Dev data format:

query[tab]pos_target1|pos_target1|...[tab]neg_target1|neg_target2|...

Parameters:

**tgt_sep**: Separator to use when splitting the targets. Default is "|".

**neg_num**: Number of negative targets for each query. Default is -1, which is to take all targets except for positive ones
as negative targets. 


##bert_base
The bert base model to use. Set **base** to use BERT-Base, or **large** to use BERT-Large. 

##do_lower_case
Whether to ignore case (**True**) or not (**False**).

##train_data_path
Training data path

##dev_data_path
Dev data path. Best model will be selected based on metric on dev data.

##max_seq_length
Max sequence length of BERT input

##n_fine_tune_layers
Number of BERT layers to fine-tune. Set 0 to fix all BERT layers, and -1 to fine-tune all BERT layers.

##learning_rate
Learning rate for Adam optimizer

##epochs
Training epochs. The best model is automatically selected.

##batch_size
Training batch size.