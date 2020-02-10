import urllib3
import logging
import json
logger = logging.getLogger(__name__)
from getpass import getpass

from khaiii import KhaiiiApi
from konlpy.tag import Komoran
# from konlpy.tag import Mecab

# If you use windows, try this.
# !pip install eunjeon
from eunjeon import Mecab

# ETRI 형태소 분석기
class ETRIMorphology:

    def __init__(self):
        self.openapiKey = self.get_apikey()
        self.url = "http://aiopen.etri.re.kr:8000/WiseNLU"
        self.requestJson = {"access_key": self.openapiKey,
                            "argument": {"text": None, "analysis_code": "morp"}}
        self.http = urllib3.PoolManager()

    @staticmethod
    def get_apikey():
        openapikey = getpass('Type OpenAPI Key :')
        return openapikey

    @staticmethod
    def _try_connect(openApiURL, requestJson):
        response = self.http.request(
            "POST", openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson))
        return response

    @staticmethod
    def _get_json_result(response):
        json_data = json.loads(response.data.decode('utf-8'))
        return json_data

    @staticmethod
    def _check_valid_connect(json_data):
        if json_data['result'] == -1:
            if 'Invalid Access Key' in json_data['reason']:
                logger.info(json_reason)
                logger.info('Please check the openapi access key.')
                sys.exit()
            return "openapi error - " + json_reason
        else:
            return True

    def do_lang(self, text):
        self.requestJson['argument']['text'] = text
        response = self._try_connect(self.url, self.requestJson)
        json_data = self._get_json_result(response)
        res = self._check_valid_connect(json_data)
        if not res:
            print(res)
            return None
        else:
            json_return_obj = json_data['return_object']
            return_result = ""
            json_sentence = json_return_obj['sentence']
            for json_morp in json_sentence:
                for morp in json_morp['morp']:
                    return_result += str(morp['lemma']) + '/' + str(morp['type']) + " "
            return return_result[:-1]

def Analyze(self, text, SEP=' + '):
    """
    KhaiiiApi의 분석 결과를 보기좋게 돌려주는 method

    USAGE;
    ```python
    from khaiii import KhaiiiApi
    khai3 = KhaiiiApi()
    khai3.analyze('아버지가방에들어가신다 왜 자꾸 거리감들게할까 내 성격 리얼...')
    >>> [<khaiii.khaiii.KhaiiiWord at 0x7f148e002710>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a75442b0>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a75441d0>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a75443c8>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a7544668>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a7544780>,
    >>>  <khaiii.khaiii.KhaiiiWord at 0x7f14a7544828>]

    setattr(khai3.__class__, 'Analyze', Analyze)
    khai3.Analze('아버지가방에들어가신다 왜 자꾸 거리감들게할까 내 성격 리얼...')
    >>> '아버지/NNG + 가/JKS + 방/NNG + 에/JKB + 들어가/VV + 시/EP + ㄴ다/EC +
         왜/MAG + 자꾸/MAG + 거리감/NNG + 들/VV + 게/EC + 하/VV + ㄹ까/EC +
         나/NP + 의/JKG + 성격/NNG + 리/NNG + 얼/IC + ../SE + ./SF'
    ```
    """
    res = self.analyze(text)
    f = lambda x: x.__str__().split('\t')[1]
    return SEP.join(list(map(f, res)))
