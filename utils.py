import json
import torch
from typing import Union

# ***************** UTILS *****************
def prepare_sent(sequence: str, word2ind: dict, device="cpu"):
    """Transform input sequence (sentance) to proper input for RNN.

    Args:
        sequence (str): input sequence
        word2ind (dict): dictionary {"word": token_number}
    Return:
        out: input for RNN
    """
    out = [word2ind["<bos>"]]
    for word in sequence.lower().split():
        out.append(word2ind.get(word, word2ind["<unk>"]))
    return torch.LongTensor(out).to(device)

def tensor2list(pytensor: Union[torch.Tensor, list]) -> list:
    """Convert tensor to list if needed"""
    if isinstance(pytensor, torch.Tensor):
        result = pytensor.cpu().detach().tolist()
        return result
    else:
        return pytensor

def list2tensor(pylist: Union[list, torch.Tensor], device="cpu") -> torch.Tensor:
    """Convert list to tensor if needed"""
    if isinstance(pylist, list):
        result = torch.LongTensor(pylist).to(device)
        return result
    else:
        return pylist
    
def decode_seq(idxs: torch.LongTensor, ind2word: dict) -> str:
    """Decode sequence from tokens to words"""
    if isinstance(idxs, torch.Tensor):
        decoded_seq = [ind2word[w.item()] for w in idxs.cpu()]
    elif isinstance(idxs, list):
        decoded_seq = [ind2word[w] for w in idxs]
    return " ".join(decoded_seq[1:])

def save_vocab(vocab: dict, filename: str) -> None:
    """Save vocabulary with token numbers"""
    with open(filename, mode='w') as file:
        json.dump(vocab, file, ensure_ascii=False, indent=4)
        print(f"File has been saved: {filename}")
        
def download_vocab(filename: str) -> dict:
    """Download vocabulary for working with trainded RNN"""
    with open(filename, mode='r') as file:
        vocab = json.load(file)
    return vocab