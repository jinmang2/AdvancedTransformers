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
