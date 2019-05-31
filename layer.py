import re
import tensorflow as tf
import tensorflow_hub as hub

class BertLayer(tf.layers.Layer):
    def __init__(self, bert_path, n_fine_tune_layers=0, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.bert_path = bert_path
        self.output_size = 768 if "768" in bert_path else 1024
        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(BertLayer, self).get_config()
        config["n_fine_tune_layers"] = self.n_fine_tune_layers
        config["bert_path"] = self.bert_path
        return config

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        if self.n_fine_tune_layers == -1:
            for var in self.bert.variables:
                self._trainable_weights.append(var)
        else:
            layers = []
            max_layer_num = -1
            for var in self.bert.variables:
                layer_id = re.findall("/layer_(\d+)/", var.name)
                if len(layer_id) > 0:
                    layers.append((int(layer_id[0]), var))
                    max_layer_num = max(max_layer_num, int(layer_id[0]))
            layers.sort(key=lambda x: x[0])
            trainable_vars = [layer[1] for layer in layers if layer[0] > max_layer_num - self.n_fine_tune_layers]

            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, return_sequence=False):
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if return_sequence:
            return self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        else:
            return self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
