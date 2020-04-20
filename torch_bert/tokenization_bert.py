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
        super().__init__()

if __name__ == '__main__':
    file_path = "E:/KorBERT/1_bert_download_001_bert_morp_pytorch/001_bert_morp_pytorch"
    vocab_file = file_path + '/vocab.korean_morp.list'
    B = BertTokenizer(vocab_file=vocab_file)
    print(B.model_input_names)
