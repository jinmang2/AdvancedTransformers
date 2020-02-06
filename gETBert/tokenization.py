import collections
import re
import unicodedata
import six
import tensorflow as tf

class BERTTokenizer:
    """End 2 End Tokenizing NLU Embedding!"""
    def __init__(self, vocab_file, do_lower_case=False, max_len=None):
        # ETRI에서 제공한 vocab file을 읽어오고
        # 역 방향의 사전을 정의한다.
        self.vocab = self._load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # End to End Tokenizer를 구축하기 위해 아래 두 Tokenizer를 할당한다.
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        # End to End Tokenizing.
        for token in self.basic_tokenizer.tokenize(text):
            # ETRI Vocab 양식에 맞게 token 끝에 '_'를 붙여준다.
            token += '_'
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @staticmethod
    def _load_vocab(vocab_file):
        # 단어 사전을 저장할 OrderedDict 객체 생성
        vocab = collections.OrderedDict()
        index = 0
        with tf.io.gfile.GFile(vocab_file, 'r') as reader:
            while True:
                # Binary Text를 unicode(utf-8)로 decode.
                token = convert_to_unicode(reader.readline())
                if not token: break
                if ((token.find('n_iters=') == 0) or
                    (token.find('max_length=') == 0)):
                    continue
                token = token.split('\t')[0]
                token = token.strip()
                # 토큰과 해당 index를 기록
                vocab[token] = index
                index += 1
        return vocab

class BasicTokenizer:

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                # 현재 input으로 '고객/NNG'와 같이 Part-of-speech가 이미
                # tagging되어있고 vocab은 '고객/NNG_'로 단어를 기록하고 있음.
                # 여기서 `lower` 메서드를 사용하면 뒤의 tagging이 소문자로
                # 변환되어 값의 비교를 못하게 되므로 이를 주석처리.
                # 아니면 do_lower_case를 False로 둬도 무방함.
#                 token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        # https://gist.github.com/Pusnow/aa865fa21f9557fa58d691a8b79f8a6d
        # 모든 음절을 정준 분해(Canonical Decomposition)시킴
        # '각'을 'ㄱ+ㅏ+ㄱ'으로 저장(출력되는 값은 동일)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                # unicode category가 "Mark, Nonspacing"일 경우 pass
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i, start_new_word = 0, True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                # 공백이면 [" "]을 추가하고 새로운 단어로 시작
                output.append([char])
                start_new_word = True
            else:
                # 공백이 아닐 경우,
                if start_new_word:
                    # 새로운 단어로 시작할 경우에 빈 리스트 추가
                    output.append([])
                # 해당 문자부터 시작하도록 start_new_word는 False로 setting.
                start_new_word = False
                # 위에 추가한 빈 리스트에 각각 character를 채워넣음
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]


    def _clean_text(self, text):
        output = [] # char을 저장할 list 생성
        for char in text:
            # 텍스트에서 Char 단위로 출력
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                # \x00이거나 �이거나 unicode cat.이 C로 시작할 경우
                # (개행문자 제외) output에 추가하지 않는다.
                continue
            if self._is_whitespace(char):
                # 공백일 경우 " "으로 output에 추가
                output.append(" ")
            else:
                # 이 외의 경우 전부 output에 추가
                output.append(char)
        # cleaning 작업을 거친 Text를 후처리하여 반환
        return "".join(output)

    # char 단위 함수들
    @staticmethod
    def _is_whitespace(char):
        if char == " " or char == '\t' or char == '\n' or char == '\r':
            # 개행문자이거나 띄어쓰기면 True 반환
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            # unicode category가 Space Seperator면 True 반환
            return True
        # 이 외의 경우 전부 False 반환
        return False

    @staticmethod
    def _is_control(char):
        if char == "\t" or char == "\n" or char == "\r":
            # 개행문자이면 False 반환
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            # unicode category가
            # Cc(Control)
            # Cf(format)
            # Co(Private Use, is 0)
            # Cs(Surrrogate, is 0)일 경우, True 반환
            return True
        # 이 외의 경우 전부 False 반환
        return False

    @staticmethod
    def _is_punctuation(char):
        # 한국어 형태소 분석기이기 때문에 공백과 같은지 여부만 반환
        return char == ' '

class WordpieceTokenizer:

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        """
        text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # max word로 설정한 글자 수를 넘길 경우, UNK 처리
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 첫번째 글자부터 천천히 vocab에 있는 단어인지 체크
                while start < end:
                    substr = "".join(chars[start:end])
                    # do_lower_case == True일 경우에
                    # 위에서 Canonical Decomposition 과정을 거쳤기 때문에
                    # 이를 다시 Composition해줘야 vocab의 단어와 비교 가능하다.
                    substr = unicodedata.normalize("NFC", substr)
                    if substr in self.vocab:
                        # 만일 해당 단어가 vocab에 있다면 해당 단어로 break
                        cur_substr = substr
                        break
                    end -= 1
                # 만일 어떠한 단어랑도 매칭되지 않았다면, (1)로 가서 [UNK] 처리
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                # 어미, 혹은 다른 사전에 있는 단어를 찾기위해 start에 end값을 할당
                start = end
            if is_bad: # --- (1)
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

# text 단위 공백 처리
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip() # 양 사이드의 공백을 제거
    if not text: # 어떠한 값도 없을 시, 빈 list를 반환
        return []
    tokens = text.split() # 공백 단위로 쪼갠 list를 반환
    return tokens

def do_lang(openapi_key, text) :
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

    requestJson = {"access_key": openapi_key,
                   "argument": {"text": text, "analysis_code": "morp"}}

    http = urllib3.PoolManager()
    response = http.request("POST", openApiURL,
                            headers={
                                "Content-Type": "application/json; charset=UTF-8"
                            }, body=json.dumps(requestJson))

    json_data = json.loads(response.data.decode('utf-8'))
    json_result = json_data["result"]

    if json_result == -1:
        json_reason = json_data["reason"]
        if "Invalid Access Key" in json_reason:
            logger.info(json_reason)
            logger.info("Please check the openapi access key.")
            sys.exit()
        return "openapi error - " + json_reason
    else:
        json_data = json.loads(response.data.decode('utf-8'))

        json_return_obj = json_data["return_object"]

        return_result = ""
        json_sentence = json_return_obj["sentence"]
        for json_morp in json_sentence:
            for morp in json_morp["morp"]:
                return_result = return_result + str(morp["lemma"]) + "/" + str(morp["type"]) + " "

        return return_result

def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            openapi_key,
                                            output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    komoran = Komoran()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example,
                                         label_list,
                                         max_seq_length,
                                         tokenizer,
                                         openapi_key,
                                         komoran=komoran)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def convert_single_example(ex_index, example,
                           label_list,
                           max_seq_length,
                           tokenizer,
                           openapi_key,
                           komoran=None):
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {label : i for (i, label) in enumerate(label_list)}

    if openapi_key is None:
        tokens_a = ' '.join(
            [i[0] + '/' + i[1]
             for i in komoran.pos(example.text_a)])
    else:
        tokens_a = do_lang(openapi_key, example.text_a)
#     if "openapi error" in tokens_a:
#         tf.logging.info("(%d--%s)" % (ex_index, tokens_a))
    tokens_a = tokenizer.tokenize(tokens_a)

    if example.text_b:
        tokens_b = do_lang(openapi_key, example.text_b)
        if "openapi error" in tokens_a:
            tf.logging.info("(%d--%s)" % (ex_index, tokens_b))
            tokens_b = ' '.join(
                [i[0] + '/' + i[1]
                 for i in Komoran().pos(example.text_b)])
        tokens_b = tokenizer.tokenize(tokens_b)

        def _truncate_seq_pair(tokens_a, tokens_b, max_length):
            while True:
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        # Account for [CLS], [SEP],and [SEP] with "-3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
    else:
        # Account for [CLS] and [SEP] with "-2"
        if len(tokens_a) > max_seq_length-2:
            tokens_a = tokens_a[:(max_seq_length-2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    # tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    # type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    # tokens:   [CLS] the dog is hairy . [SEP]
    # type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for 'type=0' and
    # 'type=1' were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not "strictly" necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # if easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    tokens_b = None
    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)

    return feature

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, use_tpu=False):
    """Creates an 'input_fn' closure to be passed to TPUEstimator."""

    name_to_features = {
        'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([], tf.int64),
        'is_real_example': tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features, use_tpu=use_tpu):
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # But in this time, we use GPU.
        if use_tpu:
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

        return example

    def input_fn(params):
        batch_size = params['batch_size']

        # For training, we want to lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
