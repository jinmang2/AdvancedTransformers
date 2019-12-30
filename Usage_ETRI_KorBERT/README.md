# ETRI KorBERT 한국어 embedding 사용하기 및 적용 예시
2주 동안 source code 하나하나 뜯어가며 삽질한 노고를 기록하고 BERT에서 해당 코드가 어떠한 역할을 하는지 논문과 비교하며 설명!

### Requirements
- ETRI에서 제공하는 model ckpt(checkpoint)와 vocab list
  - 이는 저작권 상 Git에 올릴 수 없으니 **아래 ETRI 홈페이지에서 직접 openapi를 활용하여 받도록 한다.**
  - [ETRI 학습 모델 및 데이터 제공](http://aiopen.etri.re.kr/service_dataset.php)
- Tensorflow 1.15.0
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

    <table>
      <tr>
        <td>대분류</td>
        <td>중분류</td>
        <td>대분류</td>
      </tr>
      <tr>
        <td rowspan="5">(1) 체언</td>
        <td rowspan="3">명사</td>
        <td>일반명사(NNG)</td>
      </tr>
      <tr>
        <td>고유명사(NNP)</td>
      </tr>
      <tr>
        <td>의존명사(NNB)</td>
      </tr>
      <tr>
        <td>대명사(NP)</td>
        <td>대명사(NP)</td>
      </tr>
      <tr>
        <td>수사(NR)</td>
        <td>수사(NR)</td>
      </tr>
      <tr>
        <td rowspan="5">(2) 용언</td>
        <td>동사(VV)</td>
        <td>동사(VV)</td>
      </tr>
      <tr>
        <td>형용사(VA)</td>
        <td>형용사(VA)</td>
      </tr>
      <tr>
        <td>보조용언(VX)</td>
        <td>보조용언(VX)</td>
      </tr>
      <tr>
        <td rowspan="2">지정사(VC)</td>
        <td>긍정지정사(VCP)</td>
      </tr>
      <tr>
        <td>부정지정사(VCN)</td>
      </tr>
      <tr>
        <td rowspan="5">(3) 수식언</td>
        <td rowspan="3">관형사(MM)</td>
        <td>성상 관형사(MMA)</td>
      </tr>
      <tr>
        <td>지시 관형사(MMD)</td>
      </tr>
      <tr>
        <td>수 관형사(MMN)</td>
      </tr>
      <tr>
        <td rowspan="2">부사(MA)</td>
        <td>일반부사(MAG)</td>        
      </tr>
      <tr>
        <td>접속부사(MAJ)</td>
      </tr>
      <tr>
        <td>(3) 독립언</td>
        <td>감탄사(IC)</td>
        <td>감탄사(IC)</td>
      </tr>
      <tr>
        
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
      <tr>
      </tr>
    </table>
