<link href="//maxcdn.bootstrapcdn.com/bootstrap/latest/css/bootstrap.min.css" rel="stylesheet">
<script src="//code.jquery.com/jquery.min.js"></script>
<script src="//maxcdn.bootstrapcdn.com/bootstrap/latest/js/bootstrap.min.js"></script>

# ETRI KorBERT 한국어 embedding 사용하기 및 적용 예시
2주 동안 source code 하나하나 뜯어가며 삽질한 노고를 기록하고 BERT에서 해당 코드가 어떠한 역할을 하는지 논문과 비교하며 설명!

### Requirements
- Tensorflow 1.15.0
- ETRI에서 제공하는 model ckpt(checkpoint)와 vocab list
  - 이는 저작권 상 Git에 올릴 수 없으니 **아래 ETRI 홈페이지에서 직접 openapi를 활용하여 받도록 한다.**
  - [ETRI 학습 모델 및 데이터 제공](http://aiopen.etri.re.kr/service_dataset.php)
  - ETRI에서 제공하는 버전은 총 4개이다.
    ```
    1. Pytorch + Morphology
    2. Tensorflow + Morphology
    3. Pytorch + Eojeol
    4. Tensorflow + Eojeol
    ```
  - 형태소와 어절은 input을 형태소 분석을 하고 넣어줄 것인지, 아니면 pure text 자체를 넣어줄 것인지 여부의 차이만 존재할 뿐, 큰 차이가 없다.
  - 중요한 것은 **사용하는 형태소 분석기는 TTA 표준 형태소 태그셋(TTAK.KO-11.0010/R1)에 호환되는 형태소분석기 사용**이 필요하다.
  - [한국정보통신기술협회(Telecommuication Technology Association, TTA) 형태소 태그셋](http://aiopen.etri.re.kr/data/001.형태소분석_가이드라인.pdf)

    <table class="table table-striped table-bordered" style="width:1600px;">
      <thead>
        <tr>
          <th style="width:300px" align="center">대분류</td>
          <th style="width:300px" align="center">중분류</td>
          <th style="width:1000px" align="center">대분류</td>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td rowspan="5" align="center">(1) 체언</td>
          <td rowspan="3" align="center">명사</td>
          <td align="center">일반명사(NNG)</td>
        </tr>
        <tr>
          <td align="center">고유명사(NNP)</td>
        </tr>
        <tr>
          <td align="center">의존명사(NNB)</td>
        </tr>
        <tr>
          <td align="center">대명사(NP)</td>
          <td align="center">대명사(NP)</td>
        </tr>
        <tr>
          <td align="center">수사(NR)</td>
          <td align="center">수사(NR)</td>
        </tr>
        <tr>
          <td rowspan="5" align="center">(2) 용언</td>
          <td align="center">동사(VV)</td>
          <td align="center">동사(VV)</td>
        </tr>
        <tr>
          <td align="center">형용사(VA)</td>
          <td align="center">형용사(VA)</td>
        </tr>
        <tr>
          <td align="center">보조용언(VX)</td>
          <td align="center">보조용언(VX)</td>
        </tr>
        <tr>
          <td rowspan="2" align="center">지정사(VC)</td>
          <td align="center">긍정지정사(VCP)</td>
        </tr>
        <tr>
          <td align="center">부정지정사(VCN)</td>
        </tr>
        <tr>
          <td rowspan="5" align="center">(3) 수식언</td>
          <td rowspan="3" align="center">관형사(MM)</td>
          <td align="center">성상 관형사(MMA)</td>
        </tr>
        <tr>
          <td align="center">지시 관형사(MMD)</td>
        </tr>
        <tr>
          <td align="center">수 관형사(MMN)</td>
        </tr>
        <tr>
          <td rowspan="2" align="center">부사(MA)</td>
          <td align="center">일반부사(MAG)</td>        
        </tr>
        <tr>
          <td align="center">접속부사(MAJ)</td>
        </tr>
        <tr>
          <td align="center">(4) 독립언</td>
          <td align="center">감탄사(IC)</td>
          <td align="center">감탄사(IC)</td>
        </tr>
        <tr>
          <td rowspan="9" align="center">(5) 관계언</td>
          <td rowspan="7" align="center">격조사(JK)</td>
          <td align="center">주격조사(JKS)</td>
        </tr>
        <tr>
          <td align="center">보격조사(JKC)</td>
        </tr>
        <tr>
          <td align="center">관형격조사(JKG)</td>
        </tr>
        <tr>
          <td align="center">목적격조사(JKO)</td>
        </tr>
        <tr>
          <td align="center">부사격조사(JKB)</td>
        </tr>
        <tr>
          <td align="center">호격조사(JKV)</td>
        </tr>
        <tr>
          <td align="center">인용격조사(JKQ)</td>
        </tr>
        <tr>
          <td align="center">보조사(JX)</td>
          <td align="center">보조사(JX)</td>
        </tr>
        <tr>
          <td align="center">접속조사(JC)</td>
          <td align="center">접속조사(JC)</td>
        </tr>
        <tr>
          <td rowspan="10" align="center">(6) 의존형태</td>
          <td rowspan="5" align="center">어미(EM)</td>
          <td align="center">선어말어미(EP)</td>
        </tr>
        <tr>
          <td align="center">종결어미(EF)</td>
        </tr>
        <tr>
          <td align="center">연결어미(EC)</td>
        </tr>
        <tr>
          <td align="center">명사형전성어미(ETN)</td>
        </tr>
        <tr>
          <td align="center">관형형전성어미(ETM)</td>
        </tr>
        <tr>
          <td align="center">접두사(XP)</td>
          <td align="center">체언접두사(XPN)</td>
        </tr>
        <tr>
          <td rowspan="3" align="center">접미사(XS)</td>
          <td align="center">명사파생접미사(XSN)</td>
        </tr>
        <tr>
          <td align="center">동사파생접미사(XSV)</td>
        </tr>
        <tr>
          <td align="center">형용사파생접미사(XSA)</td>
        </tr>
        <tr>
          <td align="center">어근(XR)</td>
          <td align="center">어근(XR)</td>
        </tr>
        <tr>
          <td rowspan="10" align="center">(7) 기초</td>
          <td rowspan="6" align="center">일반기호(ST)</td>
          <td align="center">마침표, 물음표, 느낌표(SF)</td>
        </tr>
        <tr>
          <td align="center">쉼표, 가운뎃점, 콜론, 빗금(SP)</td>
        </tr>  
        <tr>
          <td align="center">따옴표, 괄호표, 줄표(SS)</td>
        </tr>
        <tr>
          <td align="center">줄임표(SE)</td>
        </tr>
        <tr>
          <td align="center">붙임표(물결)(SO)</td>
        </tr>
        <tr>
          <td align="center">기타 기호(SW)</td>
        </tr>
        <tr>
          <td align="center">외국어(SL)</td>
          <td align="center">외국어(SL)</td>
        </tr>
        <tr>
          <td align="center">한자(SH)</td>
          <td align="center">한자(SH)</td>
        </tr>
        <tr>
          <td align="center">숫자(SN)</td>
          <td align="center">숫자(SN)</td>
        </tr>
        <tr>
          <td align="center">분석불능범주(NA)</td>
          <td align="center">분석불능범주(NA)</td>
        </tr>
      </tdoby>
    </table>
