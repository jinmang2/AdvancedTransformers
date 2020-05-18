```
Python 3.6.10 |Anaconda, Inc.| (default, Mar 25 2020, 23:51:54) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from khaiii import KhaiiiApi
>>> from tokenization import BertTokenizer

>>> 
>>> 
>>> def tokenize(self, text, SEP=' + '):
...     res = self.analyze(text)
...     f = lambda x: x.__str__().split('\t')[1]
...     return ' '.join(' + '.join(list(map(f, res))).split(SEP))
... 
>>> 
>>> def setmethod(cls, func, funcname=None):
...     if funcname is None:
...         funcname = func.__name__
...     setattr(cls.__class__, funcname, func)
... 
>>> 
>>> f = open('sample.txt')
>>> res = f.readlines()
>>> f.close()
>>> print(res)
['ETRI에서 한국어 BERT 언어 모델을 배포하였다.\n', '모델은 아래 예제와 같이, 입력 문장에 대해 형태소 분석한 결과를 입력으로 받습니다.\n']
>>> 
>>> 
>>> khai3 = KhaiiiApi()
>>> setmethod(khai3, tokenize)
>>> # text = '나는 우리 엄마 집에 산다'
... text = res[0].split('\n')[0]
>>> print(khai3.tokenize(text))
ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 였/EP 다/EF ./SF
>>> 
>>> 
>>> wsl_prefix = '/mnt/e'
>>> file_path1 = "E:/KorBERT/1_bert_download_001_bert_morp_pytorch/001_bert_morp_pytorch"
>>> file_path3 = "E:/KorBERT/3_bert_download_003_bert_eojeol_pytorch/003_bert_eojeol_pytorch"
>>> # vocab_file = file_path + '/vocab.korean_morp.list'
... vocab_file1 = wsl_prefix + file_path1.split(':')[1] + '/vocab.korean_morp.list'
>>> vocab_file3 = wsl_prefix + file_path3.split(':')[1] + '/vocab.korean.rawtext.list'
>>> B1 = BertTokenizer(vocab_file=vocab_file1, max_len=100000)
>>> B3 = BertTokenizer(vocab_file=vocab_file3, max_len=100000)
>>> 
>>> print(B1.tokenize(khai3.tokenize(text)))
['E', 'T', 'R', 'I/SL_', '에서/JKB_', '한국어/NNP_', 'B', 'E', 'R', 'T/SL_', '언어/NNG_', '모델/NNG_', '을/JKO_', '배포/NNG_', '하/XSV_', '였/EP_', '다/EF_', './SF_']
>>> print(B3.tokenize(text))
['E', 'T', 'R', 'I', '에서_', '한국', '어_', 'B', 'E', 'R', 'T', '_', '언', '어_', '모델', '을_', '배', '포', '하였다', '._']
>>> 
>>> B1.vocab.update({'ETRI/SL_': B1.vocab_size})
>>> B1.vocab.update({'BERT/SL_': B1.vocab_size})
>>> print(B1.tokenize(khai3.tokenize(text)))
['ETRI/SL_', '에서/JKB_', '한국어/NNP_', 'BERT/SL_', '언어/NNG_', '모델/NNG_', '을/JKO_', '배포/NNG_', '하/XSV_', '였/EP_', '다/EF_', './SF_']
>>> B1.vocab.update({'ETRI/SL': B1.vocab_size})
>>> B1.vocab.update({'BERT/SL': B1.vocab_size})
>>> print(B1.tokenize(khai3.tokenize(text)))
['ETRI/SL_', '에서/JKB_', '한국어/NNP_', 'BERT/SL_', '언어/NNG_', '모델/NNG_', '을/JKO_', '배포/NNG_', '하/XSV_', '였/EP_', '다/EF_', './SF_']
>>> 
>>> B3.vocab.update({'ETRI_': B1.vocab_size})
>>> B3.vocab.update({'BERT_': B3.vocab_size})
>>> B3.vocab.update({'ETRI_': B3.vocab_size})
>>> print(B3.tokenize(text))
['E', 'T', 'R', 'I', '에서_', '한국', '어_', 'BERT_', '언', '어_', '모델', '을_', '배', '포', '하였다', '._']
>>> B3.vocab.update({'ETRI': B3.vocab_size})
>>> print(B3.tokenize(text))
['ETRI', '에서_', '한국', '어_', 'BERT_', '언', '어_', '모델', '을_', '배', '포', '하였다', '._']
```
