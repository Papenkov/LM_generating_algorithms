import torch
import numpy as np
from typing import Union

from utils import list2tensor, tensor2list, decode_seq

# ***************** BEAM SEARCH *****************
def predict_top_k(model, idxs: Union[list, torch.Tensor], topk: int):
    """
    Args:
        model: torch model. 
        idxs (Union[list, torch.Tensor]): токены предолжения в момент времени `t`
        topk (int): кол-во поддерживаемых вариантов на каждом шаге генерации.
    Return:
        top_probs: top_k вероятностей на шаге `t+1`
        top_idxs: top_k предсказанных (максимально вероятных) слов (токенов) на шаге `t+1`
    """
    out = model(list2tensor(idxs))
    probs = torch.softmax(out[-1], dim=0) 
    
    top_probs, top_idxs = probs.topk(topk)
    
    return tensor2list(top_probs), tensor2list(top_idxs)


def update_seq(n_idxs, top_probs: Union[np.ndarray, list], resp_probs: list, resp_idxs: list, topk: int):
     # topk_idxs
    """Обновить последовательность"""
    if len(resp_probs) != 1:
        top_probs_matrix = np.repeat(top_probs, topk).reshape((len(n_idxs), topk))
        top_probs_matrix = torch.tensor(top_probs_matrix)
    else:
        top_probs_matrix = torch.tensor(top_probs) 

    mul_probs = torch.tensor(resp_probs) * top_probs_matrix

    _top_probs, _top_idxs = mul_probs.flatten().topk(topk)
    indexes = _top_idxs % topk
    # rows = _top_idxs // topk  # разные версии torch могут ругаться на такое деление, поэтому использую след. строку
    rows = torch.div(_top_idxs, topk, rounding_mode='floor')

    add_index_to_seq = []

    for _ind, _row in zip(indexes, rows):
        add_index_to_seq.append(tensor2list(n_idxs[_row]) + [resp_idxs[_row][_ind]])
        
    return add_index_to_seq, _top_probs.tolist()


def check_eos_sentances(topk_idxs, current_probs, eos_sentences: list, eos_prob: list, topk: int, word2ind: dict):
    """
    Проверка конца предложения. Если <eos> - добавляем в `eos_sentences` убираем из `topk_idxs`
    Тоже самое с вероятностями.
    """
    del_numbers = []

    for number in range(len(topk_idxs)):
        if topk_idxs[number][-1] == word2ind["<eos>"]:
            eos_sentences.append(topk_idxs[number])
            eos_prob.append(current_probs[number])
            del_numbers.append(number)
    for del_idx in del_numbers[::-1]:
        topk_idxs.pop(del_idx)
        current_probs.pop(del_idx)
        
    return topk_idxs, topk - len(del_numbers)


def choose_best_sentance(eos_sentences: list, eos_prob: list, ind2word: dict):
    """
    Выбирает последовательность в максимальной вероятнотью. 
    Из-за возможной разной длины сгенерированных предложений умножаю полученную вероятность на длину предложения.

    Args:
        eos_sentences: top@k предложений-кандидатов.
        eos_prob: вероятности top@k предложений-кандидатов.
        ind2word: словарь {token_number: "word"}.
    Return:
        best_sentance: сгенерированное предложение (токены) с наибольшей совокупной вероятностью.
    """
    sent_lens = torch.tensor([len(sent) for sent in eos_sentences]) 
    eos_probs = torch.tensor(eos_prob)
    total_probs = sent_lens * eos_probs
    vals, idxs = torch.sort(total_probs, descending=True)
    for num, idx in enumerate(idxs):
        print(f"{num+1}. {decode_seq(eos_sentences[idx], ind2word)} -> probs: {np.round(total_probs[idx].item(), 7)}")
    best_sent_ind = total_probs.argmax()
    best_sentance = eos_sentences[best_sent_ind]
    
    return best_sentance


def generate_beam_search(model, idxs: torch.Tensor, topk: int, generate_n_tokens: int, word2ind: dict, ind2word: dict):
    """
    Реализация Beam Search алгоритма для генерации текста.

    Args:
        model: RNN модель.
        idxs: входное предложение, преобразованное в токены, которое нужно продолжить.
        topk: кол-во вариантов, которое будет поддерживать алгоритм на каждой итерации.
        generate_n_tokens: максимальная длина генерируемой последовательности.
        word2ind: словарь вида {"word": token_number}
        ind2word: словарь вида {token_number: "word"}
    Return:
        best_sentance: сгенерированное предложение, с максимальной вероятностью `beam_search`
    """
    topk_idxs = [idxs]
    current_probs = np.ones(topk)
    eos_sentences, eos_prob = [], []

    # собираем topk вероятностей и индексов
    for _ in range(generate_n_tokens):
        candidates_idxs, candidates_probs = [], []

        for idxs in topk_idxs:
            new_topk_probs, new_topk_idxs = predict_top_k(model, idxs, topk)
            candidates_idxs.append(new_topk_idxs) 
            candidates_probs.append(new_topk_probs)

        # после необходимых вычислений, добавить top@k индексов в начальную последовательность
        topk_idxs, current_probs = update_seq(topk_idxs, current_probs, candidates_probs, candidates_idxs, topk)
        topk_idxs, topk = check_eos_sentances(topk_idxs, current_probs, eos_sentences, eos_prob, topk, word2ind)
        if not len(topk_idxs):
            best_sentance = choose_best_sentance(eos_sentences, eos_prob, ind2word)
            break
            
    return best_sentance