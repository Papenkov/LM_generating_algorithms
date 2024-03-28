# Реализация алгоритмов сэмплирования для RNN языковой модели

Алгоритмы сэмплирования:
- Greedy
- Top@K
- Top@P
- BeamSearch

## Inference
Для примера в папке `./data` есть обученная модель с весами и маппинг слов-индексов к ней.  
Модель обучалась на наборе отзывов на фильмы `imdb` из библиотеки `datasets`.  

Пример использования.

```python

import torch

from utils import download_vocab 
from model import LanguageModel
from main import sampling

device = "cpu"
# download model
model = torch.load("./data/gru_lm.pth", map_location=torch.device(device))
# download vocabulary mapping
word2ind = download_vocab("./data/word2ind.json")
ind2word = {v: k for k, v in word2ind.items()}

input_sentence = "I was sceptical about"
mode = "random"     # one of ["greedy", "random", "beam_search"]
topk = 7            # Top@K и Top@P реализованы в одном `mode`="random". Для Top@K тип `int`, для Top@P - `float`

generated_sentence = sampling(model, 
                              sent=input_sentence, mode=mode, 
                              word2ind=word2ind, ind2word=ind2word,
                              topk=topk)
```

## Sampling parameters

| **Argument** | **Type**      | **Description**                | **Default** |
|---------|----------------|--------------------------------|-------------|
| model    | PyTorch model            | PyTorch LM           | None      |
| sent    | str            | input sentence  | None       |
| mode     | str            | one of ["greedy", "random", "beam_search"]. "random" includes `Top@K`, `Top@P`         | None      |
| word2ind     | dict            | dictionary mapping {"word": token_number}         | None    |
| ind2word     | dict            | dictionary mapping {token_number: "word"}        | None      |
| topk     | Union[int, float]            | `int` for `Top@K` and `Beam Search` (number of samples), float for `Top@P` (total probability)        | None      |
| max_seq_len     | int            | max output sequence len         | 32     |
