# Data Format

## Ranking:

Training data format:

**query**[tab]**pos\_target1**|**pos\_target1**|...

Dev data format:

**query**[tab]**pos\_target1**|**pos\_target1**|...[tab]**neg\_target1**|**neg\_target2**|...

Parameters: `sep`, `neg_num`. See `README.md` under `example_conf`.

## Classification:

Training and dev data format:

**query**[tab]**class\_label**

Zeal will automatically convert ``class_label`` to id and record the mapping in `output/classes.txt`.

## Sequence Tagging:

Training and dev data format:

**token1**|**token1**|...[tab]**token1\_tag**|**token2\_tag**...

Zeal will automatically convert token tags to id and record the mapping in `output/tags.txt`.