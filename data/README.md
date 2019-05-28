# Data Format

## Ranking:

Training data format:

query[tab]pos_target1|pos_target1|...

Dev data format:

query[tab]pos_target1|pos_target1|...[tab]neg_target1|neg_target2|...

Parameters: `pos_sep`, `neg_num`. See `README.md` under `example_conf`.

## Classification:

Training and dev data format:

query[tab]class_label

Text label is supported. Zeal will automatically convert text label to id and record the mapping in `output/classes.txt`.