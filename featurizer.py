class BertTokenizer(object):
    '''Re-implementation of Bert tokenizer.'''
    def __init__(self, vocab_file, do_lower_case=True, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = {}
        index = 0
        for line in open(vocab_file, encoding="utf8"):
            token = line.strip()
            if token:
                self.vocab[token] = index
                index += 1

        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def wordpiece_tokenize(self, token):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return self.unk_token

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end

        if is_bad:
            return self.unk_token
        else:
            return sub_tokens

    def tokenize(self, text):
        output_tokens = []
        for token in text.strip().split():
            if self.do_lower_case:
                token = token.lower()
            sub_tokens = self.wordpiece_tokenize(token)
            output_tokens.extend(sub_tokens)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] if token in self.vocab else self.vocab[self.unk_token] for token in tokens]

class BertFeaturizer(object):
    def __init__(self, vocab_file, do_lower_case):
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
      """Truncates a sequence pair in place to the maximum length."""

      # This is a simple heuristic which will always truncate the longer sequence
      # one token at a time. This makes more sense than truncating an equal percent
      # of tokens from each, since if one sequence is very short then each token
      # that's truncated likely contains more information than a longer sequence.
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def featurize(self, text_a, text_b, max_seq_length):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
        }

    def featurize_with_tag(self, token_tags, tag2id, max_seq_length):
        """Featurize input with tagging information"""

        # token_tags: [(token1, tag1), (token2, tag2), ...]
        # tag2id: dict that transform tag to id.

        tokens, tags = [], []
        # The tag input mask has 1 for the first subtoken of real tokens and 0 for other subtokens,
        # padding tokens and [CLS], [SEP].
        tag_mask = []

        tokens.append("[CLS]")
        tags.append("[CLS]")
        tag_mask.append(0)

        for token, tag in token_tags:
            sub_tokens = self.tokenizer.wordpiece_tokenize(token)
            tokens.extend(sub_tokens)
            tags.extend([tag]*len(sub_tokens))
            tag_mask.extend([1] + [0]*(len(sub_tokens)-1))
        if len(tokens) > max_seq_length - 1:
            tokens = tokens[:max_seq_length-1]
            tags = tags[:max_seq_length-1]
            tag_mask = tag_mask[:max_seq_length-1]

        tokens.append("[SEP]")
        tags.append("[SEP]")
        tag_mask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tag_ids = [tag2id[tag] if tag in tag2id else 0 for tag in tags]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            tag_ids.append(0)
            tag_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(tag_ids) == max_seq_length
        assert len(tag_mask) == max_seq_length

        return {
            "tokens": tokens,
            "tags": tags,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "tag_ids": tag_ids,
            "tag_mask": tag_mask
        }

if __name__ == "__main__":
    vocab = "output/vocab/baseTrue.txt"
    featurizer = BertFeaturizer(vocab, True)

    print("===Test [featurize]===")
    inputs = featurizer.featurize("oppties created last year", "opportunity", 16)
    for k, v in inputs.items():
        print(k, v, len(v))

    print("===Test [featurize_with_tag]===")
    token_tags = list(zip("show|me|oppties|older|than|100|days".split("|"), "O|O|O|DATE|DATE|DATE|DATE".split("|")))
    inputs = featurizer.featurize_with_tag(token_tags, {"O":1, "DATE":2}, 16)
    for k, v in inputs.items():
        print(k, v, len(v))