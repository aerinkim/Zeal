import os
import tensorflow as tf
import numpy as np
import hashlib

class ClassificationEvaluator(tf.keras.callbacks.Callback):
    def __init__(self, config, log, featurizer):
        super(ClassificationEvaluator, self).__init__()
        self.log = log
        self.evalFile = config["dev_data_path"]
        self.featurizer = featurizer
        self.id2class = {}
        for line in open(os.path.join(log["model_path"], "classes.txt")):
            class_name, id = line.rstrip().split("\t")
            self.id2class[int(id)] = class_name

        eval_input_ids, eval_input_masks, eval_segment_ids, eval_labels = [], [], [], []
        eval_data_md5 = hashlib.md5()
        max_seq_length = int(config["max_seq_length"])
        for line in open(self.evalFile, encoding="utf8"):
            eval_data_md5.update(line.encode("utf8"))
            query, class_name = line.rstrip().split("\t")

            eval_labels.append(class_name)
            feature = featurizer.featurize(query, None, max_seq_length)
            eval_input_ids.append(feature["input_ids"])
            eval_input_masks.append(feature["input_mask"])
            eval_segment_ids.append(feature["segment_ids"])
        log["eval_data_md5"] = eval_data_md5.hexdigest()
        self.eval_samples = [eval_input_ids, eval_input_masks, eval_segment_ids, eval_labels]
        self.eval_history_file = os.path.join(log["model_path"], "eval.log")
        self.best_metric = None


    def on_epoch_end(self, epochs, logs=None):
        print()
        print("Evaluating on " + self.evalFile)
        labels = self.eval_samples[-1]
        scores = self.model.predict(self.eval_samples[:3])
        pred_labels = [self.id2class[i] for i in np.argmax(scores, -1)]

        acc = 1.0
        for i in range(len(labels)):
            if labels[i] == pred_labels[i]:
                acc += 1
        acc = acc / len(labels)
        result_str = "Epoch {0}: Accuracy {1:.2f}".format(epochs, acc)
        print(result_str)
        with open(self.eval_history_file, "a+", encoding="utf8") as f:
            f.write(result_str + "\n")

        if self.best_metric == None or self.best_metric < acc:
            self.best_metric = acc
            self.log["eval_metric"] = {"best_epoch":epochs, "accuracy": acc}
            self.model.save(os.path.join(self.log["model_path"], "model.h5"))

class RankingEvaluator(tf.keras.callbacks.Callback):
    def __init__(self, config, log, featurizer):
        super(RankingEvaluator, self).__init__()
        self.log = log
        self.evalFile = config["dev_data_path"]
        self.featurizer = featurizer
        self.targets = set()
        for line in open(os.path.join(log["model_path"], "targets.txt")):
            self.targets.add(line.rstrip())

        tgt_sep = "|"
        if "tgt_sep" in config:
            tgt_sep = config["tgt_sep"]

        self.eval_samples = []
        eval_data_md5 = hashlib.md5()
        max_seq_length = int(config["max_seq_length"])
        for line in open(self.evalFile, encoding="utf8"):
            eval_data_md5.update(line.encode("utf8"))
            eval_input_ids, eval_input_masks, eval_segment_ids, eval_labels = [], [], [], []
            query, pos_tgts, neg_tgts = line.rstrip().split("\t")
            pos_cols = set()
            for pos_col in set(pos_tgts.split(tgt_sep)):
                eval_labels.append(1)
                feature = featurizer.featurize(query, pos_col, max_seq_length)
                eval_input_ids.append(feature["input_ids"])
                eval_input_masks.append(feature["input_mask"])
                eval_segment_ids.append(feature["segment_ids"])
            for neg_col in set(neg_tgts.split(tgt_sep)) - pos_cols:
                eval_labels.append(0)
                feature = featurizer.featurize(query, neg_col, max_seq_length)
                eval_input_ids.append(feature["input_ids"])
                eval_input_masks.append(feature["input_mask"])
                eval_segment_ids.append(feature["segment_ids"])
            self.eval_samples.append([eval_input_ids, eval_input_masks, eval_segment_ids, eval_labels])
        log["eval_data_md5"] = eval_data_md5.hexdigest()
        self.eval_history_file = os.path.join(log["model_path"], "eval.log")
        self.best_metric = None

    def precision_recall_f1(self, labels, preds):
        TP, pred_one, true_one = 0, 0, 0
        for p, l in zip(preds, labels):
            if p == 1 and l == 1:
                TP += 1
            if p == 1:
                pred_one += 1
            if l == 1:
                true_one += 1
        precision = TP * 1.0 / pred_one if pred_one > 0 else 1.0
        recall = TP * 1.0 / true_one if true_one > 0 else 1.0
        F1 = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, F1

    def on_epoch_end(self, epochs, logs=None):
        print()
        print("Evaluating on " + self.evalFile)
        label_scores = []
        for sample in self.eval_samples:
            labels = sample[-1]
            scores = [x[0] for x in self.model.predict(sample[:3])]
            label_scores.append((labels, scores))

        n_samples = len(label_scores)
        result = []
        for threshold in np.arange(0.01, 1.0, 0.01):
            accuracy, precision, recall, F1 = 0.0, 0.0, 0.0, 0.0
            for label_score in label_scores:
                pred_labels = [1 if score > threshold else 0 for score in label_score[1]]
                p, r, f = self.precision_recall_f1(label_score[0], pred_labels)
                accuracy += np.sum(np.array(label_score[0]) == np.array(pred_labels)) * 1.0 / len(pred_labels)
                precision += p
                recall += r
                F1 += f
            result.append(
                {
                    "epoch": epochs,
                    "threshold": threshold,
                    "precision": precision / n_samples,
                    "recall": recall / n_samples,
                    "F1": F1 / n_samples,
                    "accuracy": accuracy / n_samples,
                })

        result = sorted(result, key=lambda x: x["F1"], reverse=True)[0]
        result_str = "Epoch {5}: Best threshold={0:.2f}, precision={1:.2f}, recall={2:.2f}, F1={3:.2f}, Accuracy={4:.2f}".format(
            result["threshold"],
            result["precision"],
            result["recall"],
            result["F1"],
            result["accuracy"],
            result["epoch"])
        print(result_str)
        with open(self.eval_history_file, "a+", encoding="utf8") as f:
            f.write(result_str + "\n")

        if self.best_metric == None or self.best_metric["F1"] < result["F1"]:
            self.best_metric = {k:v for k,v in result.items()}
            best_eval_metric = {}
            for k, v in result.items():
                if k == "epoch":
                    best_eval_metric["best_epoch"] = v
                else:
                    best_eval_metric[k] = v
            self.log["eval_metric"] = best_eval_metric
            self.model.save(os.path.join(self.log["model_path"], "model.h5"))
