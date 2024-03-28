import torch

from utils import prepare_sent, decode_seq
from beam_search import generate_beam_search

# ***************** MAIN FUNCTIONS *****************
def generate(model, 
             idxs: torch.LongTensor, 
             mode: str, 
             word2ind: dict,
             topk: int = None, 
             min_single_p: float = 0.05,
             device="cpu"):
    
    out = model(idxs)
    next_word_probs = torch.softmax(out[-1], dim=0)  # probs
    
    # greedy approach
    if mode == "greedy":
        probabilities, topk_idxs = torch.topk(next_word_probs, k=1)
        idxs = torch.cat((idxs, topk_idxs), dim=0)
    
    # random samling from topk words
    elif mode == "random":
        if isinstance(topk, int):       # top_k approach  (`k` - number of samples)
            probabilities, top_idxs = torch.topk(next_word_probs, k=topk)

        elif isinstance(topk, float):   # top_p approach  (`p` - total probability)
            sorted_probs, top_idxs = next_word_probs.sort(descending=True)  
            most_valuable_probs = torch.cumsum(sorted_probs, dim=0) < topk   # cumsum of probs
            probabilities = sorted_probs[most_valuable_probs]
            
        # delete probs under `min_single_p`
        above_min_probs = probabilities[probabilities > min_single_p] 
        candidates = top_idxs[:len(above_min_probs)]
        
        if len(candidates) == 0:
            idxs = torch.cat((idxs, torch.tensor(word2ind["<eos>"])[None].to(device)), dim=0)
        else:
            # random sampling 
            random_idx = torch.randperm(len(candidates))[0].item()
            idxs = torch.cat((idxs, top_idxs[random_idx][None]), dim=0)
    
    return idxs


def sampling(model, sent: str, mode: str, word2ind: dict, ind2word: dict,
             topk: int = None, max_seq_len = 32):
    
    mode_options = ["greedy", "random", "beam_search"]
    assert mode in mode_options, f"Invalid mode: `{mode}`, choose one of {mode_options}"

    idxs = prepare_sent(sent, word2ind)
    generate_n_tokens = max_seq_len - len(idxs) + 1  # +1 regarding to "<bos>" token

    asrt_msg = f"Invalid `max_seq_len` value in `sampling` function. `max_seq_len` has to be greater that input words sequence. \
                \n\tNow input sequence len is {len(idxs)} (function token has been added '<bos>' at the beggining) and max_seq_len {max_seq_len}"
    assert generate_n_tokens > 0, asrt_msg
    print(f"********** Mode is: `{mode}`, max words will be added: `{generate_n_tokens}` **********")

    if mode == "greedy" or mode == "random":
        if mode == "random":
            strategy = "top@K" if isinstance(topk, int) else "top@P"
            print(f"********** Generating strategy is: `{strategy}` **********")
        for _ in range(generate_n_tokens):
            if len(idxs) == max_seq_len or idxs[-1].item() == word2ind["<eos>"]:
                break
            else:
                idxs = generate(model, idxs, mode=mode, word2ind=word2ind, topk=topk)
            
    elif mode == "beam_search":
        idxs = generate_beam_search(model, idxs, topk, generate_n_tokens, word2ind, ind2word)
        
    return decode_seq(idxs, ind2word)

