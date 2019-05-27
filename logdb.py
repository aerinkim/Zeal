import json
import os
import sys
import pandas as pd

class LogDB(object):
    def __init__(self, work_path):
        self.records = []
        self.work_path = work_path
        for record_file in os.listdir(work_path):
            try:
                record = json.load(open(os.path.join(work_path, record_file), encoding="utf8"))
                self.records.append(record)
            except:
                print("WARNING: log file {0} is broken".format(record_file))
                pass

    def save_log(self, model_name, config, log):
        json.dump(
            {
                "model_name": model_name,
                "config": config,
                "log": log
            },
            open(os.path.join(self.work_path, model_name + ".json"), "w", encoding="utf8"))

    def major_view(self):
        """Extract major information from logs"""

        all_keys = set()
        major_records = []
        for record in self.records:
            major_record = {}
            major_record["model_name"] = record["model_name"]
            config = record["config"]
            for k, v in config.items():
                major_record[k] = v

            log = record["log"]
            for k in ["model_path", "train_data_md5", "eval_data_md5"]:
                major_record[k] = log[k]
            metric = record["log"]["eval_metric"]
            for k, v in metric.items():
                major_record[k] = v
            major_records.append(major_record)
            all_keys = all_keys.union(major_record.keys())

        view = {k:[] for k in all_keys}
        for major_record in major_records:
            for k in all_keys:
                if k in major_record:
                    view[k].append(major_record[k])
                else:
                    view[k].append(None)

        data = pd.DataFrame(data=view)
        key_cols = [
            "model_name",
            "problem_type",
            "train_data_md5",
            "eval_data_md5",
            "bert_base",
            "do_lower_case"] + list(record["log"]["eval_metric"].keys())
        cols = key_cols + [c for c in all_keys if c not in key_cols]
        data = data[cols]
        return data

    def export_csv(self, path):
        data = self.major_view()
        csvfile = os.path.join(path, "model.csv")
        data.to_csv(csvfile, index=False)
        print("model view exported to {0}".format(os.path.abspath(csvfile)))


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    logdb = LogDB(os.path.join(script_path, ".logdb"))
    logdb.export_csv(script_path)