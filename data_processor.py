import hashlib
import os
import numpy as np

def load_classification_data(config, log, featurizer):
    model_path = log["model_path"]
    train_data_path = config["train_data_path"]
    max_seq_length = int(config["max_seq_length"])

    #Get all classes
    class2id = {}
    class_idx = 0
    train_input_ids, train_input_masks, train_segment_ids, train_labels = [], [], [], []
    train_data_md5 = hashlib.md5()
    cnt = 0
    for line in open(train_data_path, encoding="utf8"):
        train_data_md5.update(line.encode("utf8"))
        query, class_name = line.rstrip().split("\t")
        if class_name not in class2id:
            class2id[class_name] = class_idx
            class_idx += 1

        train_labels.append(class2id[class_name])
        feature = featurizer.featurize(query, None, max_seq_length)
        train_input_ids.append(feature["input_ids"])
        train_input_masks.append(feature["input_mask"])
        train_segment_ids.append(feature["segment_ids"])

        cnt += 1
        if cnt > 0 and cnt % 1000 == 0:
            print("{0} training samples processed".format(cnt))

    #Get md5 for training file
    log["train_data_md5"] = train_data_md5.hexdigest()
    with open(os.path.join(model_path, "classes.txt"), "w", encoding="utf8") as f:
        for class_name in class2id:
            f.write(class_name + "\t" + str(class2id[class_name]) + "\n")
    config["class_num"] = len(class2id)

    return train_input_ids, train_input_masks, train_segment_ids, train_labels

def load_ranking_data(config, log, featurizer):
    model_path = log["model_path"]
    train_data_path = config["train_data_path"]
    max_seq_length = int(config["max_seq_length"])
    tgt_sep = "|"
    if "tgt_sep" in config:
        tgt_sep = config["tgt_sep"]

    neg_num = -1
    if "neg_num" in config:
        neg_num = int(config["neg_num"])

    #Get all targets
    targets = set()
    for line in open(train_data_path, encoding="utf8"):
        _, pos_tgts = line.rstrip().split("\t")
        targets |= set(pos_tgts.split(tgt_sep))
    with open(os.path.join(model_path, "targets.txt"), "w", encoding="utf8") as f:
        for target in targets:
            f.write(target + "\n")

    train_input_ids, train_input_masks, train_segment_ids, train_labels = [], [], [], []
    train_data_md5 = hashlib.md5()
    cnt = 0
    for line in open(train_data_path, encoding="utf8"):
        train_data_md5.update(line.encode("utf8"))
        query, pos_tgts = line.rstrip().split("\t")
        pos_tgts = set(pos_tgts.split(tgt_sep))
        for pos_col in pos_tgts:
            train_labels.append(1)
            feature = featurizer.featurize(query, pos_col, max_seq_length)
            train_input_ids.append(feature["input_ids"])
            train_input_masks.append(feature["input_mask"])
            train_segment_ids.append(feature["segment_ids"])

        neg_tgts = list(targets - pos_tgts)
        if neg_num > 0 and neg_num < len(neg_tgts):
            neg_tgts = list(np.random.permutation(neg_tgts)[:neg_num])

        for neg_col in neg_tgts:
            train_labels.append(0)
            feature = featurizer.featurize(query, neg_col, max_seq_length)
            train_input_ids.append(feature["input_ids"])
            train_input_masks.append(feature["input_mask"])
            train_segment_ids.append(feature["segment_ids"])
        cnt += 1
        if cnt > 0 and cnt % 1000 == 0:
            print("{0} training samples processed".format(cnt))

    #Get md5 for training file
    log["train_data_md5"] = train_data_md5.hexdigest()

    return train_input_ids, train_input_masks, train_segment_ids, train_labels