# https://mrcoding.tistory.com/entry/아톰에서-파이썬-스크립트-실행시-한글-깨짐현상-잡는-꿀팁
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# 형태소분석 기반 BERT를 위한 Tokenization Class
# 수정: joonho.lim
# 일자: 2019-05-23
#
#
# Morph와 Eojeol 버전 통합
# 수정: MyungHoon.jin
# 일자: 2020-04-20

import collections
import logging
import os
import unicodedata
from typing import List, Optional

from tokenization_utils import PretrainedTokenizer

# Huggingface 소스 파일
# VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}
#
# PRETRAINED_VOCAB_FILES_MAP = {
#     "vocab_file": {
#         "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
#         "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
#         "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
#         "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
#         "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
#         "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
#         "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
#         "bert-base-german-cased": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
#         "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
#         "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
#         "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
#         "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
#         "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
#         "bert-base-german-dbmdz-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt",
#         "bert-base-german-dbmdz-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-vocab.txt",
#         "bert-base-finnish-cased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/vocab.txt",
#         "bert-base-finnish-uncased-v1": "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/vocab.txt",
#         "bert-base-dutch-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/vocab.txt",
#     }
# }
#
# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
#     "bert-base-uncased": 512,
#     "bert-large-uncased": 512,
#     "bert-base-cased": 512,
#     "bert-large-cased": 512,
#     "bert-base-multilingual-uncased": 512,
#     "bert-base-multilingual-cased": 512,
#     "bert-base-chinese": 512,
#     "bert-base-german-cased": 512,
#     "bert-large-uncased-whole-word-masking": 512,
#     "bert-large-cased-whole-word-masking": 512,
#     "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
#     "bert-large-cased-whole-word-masking-finetuned-squad": 512,
#     "bert-base-cased-finetuned-mrpc": 512,
#     "bert-base-german-dbmdz-cased": 512,
#     "bert-base-german-dbmdz-uncased": 512,
#     "bert-base-finnish-cased-v1": 512,
#     "bert-base-finnish-uncased-v1": 512,
#     "bert-base-dutch-cased": 512,
# }
#
# PRETRAINED_INIT_CONFIGURATION = {
#     "bert-base-uncased": {"do_lower_case": True},
#     "bert-large-uncased": {"do_lower_case": True},
#     "bert-base-cased": {"do_lower_case": False},
#     "bert-large-cased": {"do_lower_case": False},
#     "bert-base-multilingual-uncased": {"do_lower_case": True},
#     "bert-base-multilingual-cased": {"do_lower_case": False},
#     "bert-base-chinese": {"do_lower_case": False},
#     "bert-base-german-cased": {"do_lower_case": False},
#     "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
#     "bert-large-cased-whole-word-masking": {"do_lower_case": False},
#     "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
#     "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
#     "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
#     "bert-base-german-dbmdz-cased": {"do_lower_case": False},
#     "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
#     "bert-base-finnish-cased-v1": {"do_lower_case": False},
#     "bert-base-finnish-uncased-v1": {"do_lower_case": True},
#     "bert-base-dutch-cased": {"do_lower_case": False},
# }

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file, encoding="utf-8"):
    vocab = collections.OrderedDict()
    index = 0
    # huggingface 코드에서는 단순하게 `.readlines()` 메서드로 구현
    with open(vocab_file, "r", encoding=encoding) as reader:
        while True:
            token = reader.readline()
            # token = convert_to_unicode(token)
            if not token:
                break
            # ETRI Vocab을 위한 코드
            if token.find('n_iters=') == 0 or token.find('max_length=') == 0:
                continue
            # index 1은 빈도수, 빈도수가 제일 높은 token부터 numbering
            token = token.split('\t')[0].strip()
            vocab[token] = index
            index += 1
    return vocab


# text 단위 공백 처리
def whitespace_tokenize(text):
    """Run basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(PretrainedTokenizer):

    vocab_file_names = VOCAB_NAME
    pretrained_vocab_files_map = PRETRAINED_VOCAB_ARCHIVE_MAP
    # pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP

    def __init__(self,
                 vocab_file,
                 do_lower_case=False,
                 do_basic_tokenize=True,
                 never_split=None,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]",
                 **kwargs):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
            )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, token) for token, ids in self.vocab.items()]
        )
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab,
            unk_token=self.unk_token
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        # added_tokens_encoder는 추가할 때 필요, default == {}
        return dict(self.vocab, **self.added_tokens_encoder)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                token += '_' # ETRI BERT에서의 차이점.
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def build_inputs_with_special_tokens(self,
        token_ids_0: List[int], token_ids_1: Optional[List[int]]=None
        ) -> List[int]:
        """
        sequence 분류 task를 위한 model input build!
        - single sequence: ``[CLS] A [SEP]``
        - pair of sequence: ``[CLS] A [SEP] B [SEP]``
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self,
        token_ids_0: List[int], token_ids_1: Optional[List[int]]=None,
        already_has_special_tokens: bool=False) -> List[int]:
        """
        special token이 추가되지 않은 list에서 sequence ids를 검색
        ``prepare_for_model``, ``encode_plus`` 메서드로 special tokens을
        추가할 때 호출됨
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self,
        token_ids_0: List[int], token_ids_1: Optional[List[int]]=None
        ) -> List[int]:
        """
        sequence pair 분류 문제를 위해 concat mask를 생성

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

            만일 token_ids_1이 None이면 0으로 채워진 mask를 반환
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [0]

    def save_vocabulary(self, vocab_path):
        pass


class BasicTokenizer:

    """Run basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=False, never_split=[],
                 tokenize_chinese_chars=True):
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=[]):
        never_split = self.never_split + never_split
        text = self._clean_text(text)
        # Chinese Char은 무시한다.
        orig_token = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                # 형태소 분석기를 사용할 경우 do_lower_case를 False로 설정할 것.
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_token))
        return output_tokens

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = [] # char을 저장한 list 생성
        for char in text:
            # 텍스트에서 char 단위로 출력
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
        # cleaning 작업을 거친 text를 후처리하여 반환
        return "".join(output)

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        # https://gist.github.com/Pusnow/aa865fa21f9557fa58d691a8b79f8a6d
        # 모든 음절을 정준 분해(Canonical Decomposition)시킴
        # `각`을 `ㄱ+ㅏ+ㄱ`으로 저장(출력되는 값은 동일)
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
        # 근데 사실상 whitespacing을 하고 ETIR가 _is_punctuation 함수를
        # 띄어쓰기만 검색하도록 만들어놔서 사실 의미없음 ㅇㅅㅇ
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i, start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                # 구두점일 경우  [char}을 추가하고 새로운 단어로 시작
                output.append([char])
                start_new_word = True
            else:
                # 구두점이 아닐 경우
                if start_new_word:
                    # 새로운 단어로 시작할 경우에 빈 리스트 추가
                    output.append([])
                # 해당 문자부터 시작하도록 start_new_word는 False로 setting
                start_new_word = False
                # 위에 추가한 빈 리스트에 각각 character를 채워넣음
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    # char 단위 함수들 ------------------------------------------------------
    @staticmethod
    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character"""
        # \t, \n, \r은 technically control characters지만
        # whiteapce로 여기고 이를 처리
        if char == " " or char == '\t' or char == '\n' or char == '\r':
            return True
        cat = unicodedata.category(char)
        if cat == 'Zs':
            # unicode category가 Space Seperator면 True 반환
            return True
        # 이 외의 경우 전부 False 반환
        return False

    @staticmethod
    def _is_control(char):
        """Checks whether `chars` is a control character"""
        if char == "\t" or char == "\n" or char == "\r":
            # \t, \n, \r을 우리는 whitespace로 처리함
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
        """Checks whether `chars` is a punctuatoin character."""
        # 왜 때문인지 모르겠지만 ETRI에서 아래부분을 주석처리해버림
        # 구두점을 띄어쓰기만 고려? 흠...
        return char == ' '

        cp = ord(char)
        # 모든 non-letter/number ASCII를 구두점으로 처리
        # "^", "$", "`"와 같은 char은 unicode에 없음
        # 그러나 이를 일관성있게 punctuation으로 처리하기 위해 아래와 같이 처리
        if ((cp >= 33 and cp <= 47) or
            (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or
            (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False


class WordpieceTokenizer:

    """Runs WordPiece tokenization"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        greedy longest-match-first algorithm을 사용하여
        주어진 vocab으로 tokenization을 수행

        20.04.20
        - 여기에 기능 추가해야함!! -> 없는 토큰 추가 학습하도록
        - 미리 빼둬야함!!
        """
        # text = convert_to_unicode(text)
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # max word로 설정한 글자 수를 넘길 경우 [UNK] 처리
                output.tokens.append(self.unk_token)
                continue
            is_bad, start = False, 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 첫 번째 글자부터 천천히 vocab에 있는 단어인지 체크
                # 맨 처음에는 해당 token자체가 이미 있는지 체크! (때문에 longest)
                while start < end:
                    substr = "".join(chars[start:end])
                    # Canonical Decomposition 과정을 거쳤기 때문에
                    # 이를 다시 Composition해줘야 vocab의 단어와 비교 가능
                    substr = unicodedata.normalize("NFC", substr)
                    #
                    # if start > 0:
                    #     substr = "##" + substr
                    if substr in self.vocab:
                        # 만일 해당 단어가 vocab에 있으면 해당 단어로 break
                        cur_substr = substr
                        break
                    end -= 1
                # 만일 어떠한 단어랑도 매칭되지 않았다면 (1)로 가서 [UNK] 처리
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                # 어미, 혹은 다른 사전에 있는 단어를 찾기 위해 start에 end값 할당
                start = end
            if is_bad: # --- (1)
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


if __name__ == '__main__':

    file_path = "E:/KorBERT/1_bert_download_001_bert_morp_pytorch/001_bert_morp_pytorch"
    vocab_file = file_path + '/vocab.korean_morp.list'
    B = BertTokenizer(vocab_file=vocab_file, max_len=100000)
    print(B.unk_token)
    print(B.all_special_tokens)
    print(B.max_len)
    print(B.vocab[B.unk_token])
    print(B._convert_token_to_id('다/EF_'))
    print(B.cls_token_id, B.sep_token_id)
    print(B.cls_token)
    print(B.vocab['모란/NNG_'])
