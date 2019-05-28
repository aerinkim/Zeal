# Data Format

## Ranking:

Training data format:

**query**[tab]**pos\_target1**|**pos\_target1**|...

Dev data format:

**query**[tab]**pos\_target1**|**pos\_target1**|...[tab]**neg\_target1**|**neg\_target2**|...

Parameters: `pos_sep`, `neg_num`. See `README.md` under `example_conf`.

## Classification:

Training and dev data format:

**query**[tab]**class\_label**

``class_label`` can be text, in which case Zeal will automatically convert text label to id and record the mapping in `output/classes.txt`.