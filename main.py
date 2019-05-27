import argparse
import os
import sys
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import hashlib
import layer
import numpy as np
from featurizer import BertFeaturizer
from evaluator import RankingEvaluator
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from logdb import LogDB

parser = argparse.ArgumentParser(description='Zero-effort and Agile Learning Tool')
parser.add_argument("job", type=str, choices=["train", "eval", "predict"],
                    help="job can be train, eval or predict")
parser.add_argument("--conf", help="conf file path")
args = parser.parse_args()

bert_paths = {
    ("base", True): "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
    ("large", True): "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1",
    ("base", False): "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1",
    ("large", False): "https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1",
}

def read_conf(conf_path):
    config = {}
    for line in open(conf_path, encoding="utf8"):
        if line.strip() == "":
             continue
        fields = line.strip().split("\t")
        config[fields[0]] = fields[1]
    config["train_data_path"] =  os.path.abspath(config["train_data_path"])
    config["dev_data_path"] =  os.path.abspath(config["dev_data_path"])

    return config

def load_training_data(config, log, featurizer):
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
            print("training data {0} processed".format(cnt))

    #Get md5 for training file
    log["train_data_md5"] = train_data_md5.hexdigest()

    return train_input_ids, train_input_masks, train_segment_ids, train_labels

def build_model(config, log):
    bert_path = log["bert_path"]
    n_fine_tune_layers = int(config["n_fine_tune_layers"])
    max_seq_length = int(config["max_seq_length"])

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype=tf.int32)
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks", dtype=tf.int32)
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids", dtype=tf.int32)
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = layer.BertLayer(bert_path, n_fine_tune_layers)(bert_inputs)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


class loss_history(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(float(logs.get('loss')))

if args.job == "train":
    log = {}
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    log_db_path = os.path.join(script_path, ".logdb")
    if not os.path.exists(log_db_path):
        os.mkdir(log_db_path)
    logdb = LogDB(log_db_path)

    output_path = os.path.join(script_path, "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    log["model_path"] = os.path.abspath(model_path)

    vocab_path = os.path.join(output_path, "vocab")
    if not os.path.exists(vocab_path):
        os.mkdir(vocab_path)

    conf_path = os.path.abspath(args.conf)
    config = read_conf(conf_path)
    bert_base = config["bert_base"]
    do_lower_case = True if config["do_lower_case"] == "True" else False
    bert_path = bert_paths[(bert_base, do_lower_case)]
    log["bert_path"] = bert_path

    with tf.keras.backend.get_session() as sess:
        print("Getting Bert Module...")
        bert_module = hub.Module(bert_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file = sess.run(tokenization_info["vocab_file"])
        output_vocab_file = os.path.join(vocab_path, bert_base + str(do_lower_case) + ".txt")
        if not os.path.exists(output_vocab_file):
            shutil.copyfile(vocab_file, output_vocab_file)

        featurizer = BertFeaturizer(vocab_file, do_lower_case)
        train_data = load_training_data(config, log, featurizer)
        rankingEvaluator = RankingEvaluator(config, log, featurizer)
        model = build_model(config, log)
        optimizer = optimizers.Adam(float(config["learning_rate"]))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        # Instantiate variables
        initialize_vars(sess)

        history = loss_history()
        model.fit(
            train_data[:3],
            train_data[-1],
            callbacks=[rankingEvaluator, history],
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"])
        )
    log["train_loss"] = history.losses
    logdb.save_log(model_name, config, log)

    # tf.saved_model.simple_save(
    #     sess,
    #     export_path,
    #     inputs={'input_image': model.input},
    #
    #     outputs={t.name: t for t in model.outputs})
