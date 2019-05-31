import argparse
import os
import sys
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import hashlib
import layer
import data_processor
import evaluator
import numpy as np
from featurizer import BertFeaturizer
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from logdb import LogDB

parser = argparse.ArgumentParser(description='Zero-effort Automatic Learning Tool')
parser.add_argument("job", type=str, choices=["train", "eval", "predict", "export"],
                    help="job can be train, eval, predict or export")
parser.add_argument("--conf", help="conf file path")
parser.add_argument("--note", help="notes on data and model settings")
parser.add_argument("--model_path", help="trained model folder path (used in eval, predict and export mode)")

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

def build_model(config, log):
    bert_path = log["bert_path"]
    n_fine_tune_layers = int(config["n_fine_tune_layers"])
    max_seq_length = int(config["max_seq_length"])

    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids", dtype=tf.int32)
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks", dtype=tf.int32)
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids", dtype=tf.int32)
    bert_inputs = [in_id, in_mask, in_segment]

    if config["problem_type"] == "point_wise_ranking":
        bert_output = layer.BertLayer(bert_path, n_fine_tune_layers)(bert_inputs)
        pred = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    elif config["problem_type"] == "classification":
        bert_output = layer.BertLayer(bert_path, n_fine_tune_layers)(bert_inputs)
        pred = tf.keras.layers.Dense(config["class_num"], activation='softmax')(bert_output)
    elif config["problem_type"] == "sequence_tag":
        bert_output = layer.BertLayer(bert_path, n_fine_tune_layers)(bert_inputs, return_sequence=True)
        pred = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config["tag_num"], activation='softmax'))(bert_output)
    else:
        raise Exception("No model is supported for problem_type {0}".format(config["problem_type"]))

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
    log["note"] = args.note
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
    shutil.copyfile(conf_path, os.path.join(model_path, "model.conf"))

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
        optimizer = optimizers.Adam(float(config["learning_rate"]))
        history = loss_history()
        model = build_model(config, log)

        if config["problem_type"] == "point_wise_ranking":
            train_data = data_processor.load_ranking_data(config, log, featurizer)
            rankingEvaluator = evaluator.RankingEvaluator(config, log, featurizer)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            model.summary()
            # Instantiate variables
            initialize_vars(sess)

            model.fit(
                train_data[:3],
                train_data[-1],
                callbacks=[rankingEvaluator, history],
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"])
            )
        elif config["problem_type"] == "classification":
            train_data = data_processor.load_classification_data(config, log, featurizer)
            classificationEvaluator = evaluator.ClassificationEvaluator(config, log, featurizer)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
            model.summary()
            # Instantiate variables
            initialize_vars(sess)

            model.fit(
                train_data[:3],
                to_categorical(train_data[-1], config["class_num"]),
                callbacks=[classificationEvaluator, history],
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"])
            )
        elif config["problem_type"] == "sequence_tag":
            train_data = data_processor.load_tagging_data(config, log, featurizer)
            sequenceTagEvaluator = evaluator.SequenceTagEvaluator(config, log, featurizer)
            model.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
                sample_weight_mode="temporal"
            )
            model.summary()
            # Instantiate variables
            initialize_vars(sess)

            target = np.array([to_categorical(x, config["tag_num"]) for x in train_data[3]])
            sample_weight = np.array(train_data[4])
            model.fit(
                train_data[:3],
                target,
                sample_weight = sample_weight,
                callbacks=[sequenceTagEvaluator, history],
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"])
            )
        else:
            raise Exception("problem_type is not recognized")
        print("Best model has been saved to {0}".format(os.path.abspath(log["model_path"])))

    log["train_loss"] = history.losses
    logdb.save_log(model_name, config, log)

elif args.job == "export":
    if args.model_path is None:
        raise Exception("Please specify model path")

    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    model = tf.keras.models.load_model(os.path.join(args.model_path, "model.h5"), custom_objects={'BertLayer': layer.BertLayer})
    export_path = os.path.join(args.model_path, "serve")
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={t.name: t for t in model.input},
            outputs={t.name: t for t in model.outputs})
    print("Servable model has been exported to {0}".format(os.path.abspath(export_path)))

else:
    raise Exception("Job type {0} is not supported for now".format(args.job))