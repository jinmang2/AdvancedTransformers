# ETRI Pytorch version BERT code

#### 20.04.20 (월)
- `huggingface.tokenizers`는 `Rust`로 작성
- 때문에 Etri에서 제공한 Wordpiece Tokenizer는 직접 구현한 것으로 추정됨
- 아니면 rust code를 python으로 포팅했거나
- 혹은 tensorflow에서 사용한 version의 코드이거나
- 아니네 이미 있네! fast version이냐 python이냐 차인가? 살펴보자
- 추가적인 코드 작성할 필요 있음!! 없는 token!
