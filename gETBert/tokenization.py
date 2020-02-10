# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
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
# 주석 및 새롭게 코드 수정
# 작성자: MyungHoon Jin

import collections
import re
import unicodedata
import six
import tensorflow as tf

def convert_to_unicode(text):
    # Python version이 3.x일 때,
    # type(text)이 `bytes`일 경우, utf-8로 변환
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    # Python version이 2.x일 때,
    # type(text)이 `str`일 경우, utf-8로 변환
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    # Python 3.x, 2.x만 허용!
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def printable_text(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class BERTTokenizer:
    """End 2 End Tokenizing NLU Embedding!"""
    # from_pretrained method는 향후 추가!
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
        ids = _convert_by_vocab(self.vocab, tokens)
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len))
        return ids

    def convert_ids_to_tokens(self, ids):
        return _convert_by_vocab(self.inv_vocab, ids)

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

    @staticmethod
    def _convert_by_vocab(vocab, items):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            output.append(vocab[item])
        return output

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

                # token.lower()

                # 모든 음절을 정준 분해시키는 함수
                token = self._run_strip_accents(token)
            # whitespacing이랑 다를게 무엇인지?
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, token):
        """Strips accents from a piece of text."""
        token = unicodedata.normalize("NFD", token)
        # https://gist.github.com/Pusnow/aa865fa21f9557fa58d691a8b79f8a6d
        # 모든 음절을 정준 분해(Canonical Decomposition)시킴
        # '각'을 'ㄱ+ㅏ+ㄱ'으로 저장(출력되는 값은 동일)
        output = []
        for char in token:
            cat = unicodedata.category(char)
            if cat == "Mn":
                # unicode category가 "Mark, Nonspacing"일 경우 pass
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, token):
        """Splits punctuation on a piece of text."""
        chars = list(token)
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
            # https://www.compart.com/en/unicode/category/Zs
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
            # https://en.wikipedia.org/wiki/Control_character
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
