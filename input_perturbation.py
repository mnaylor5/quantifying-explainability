import numpy as np
import torch
import re

DEFAULT_SAMPLE_TOP_K = 10
DEFAULT_REPLACEMENT_PROB = 0.1

## ----- Embedding-based replacement for transformers ----- ##
def construct_embedding_distance_matrix(embedding_layer, p=2.0):
    '''
    Utility to construct a pairwise distance matrix between entries in 
    an embedding weight matrix. 
    
    Input: embedding_layer; a PyTorch module of the specific embeddings
      to use (e.g.: bigbird.bert.embeddings.word_embeddings). Size is
      given as Vocab x Dim
      
    Output: distance_matrix; a torch.Tensor with shape Vocab x Vocab, whose
      entries `ij` are the p-norm distance between vocab entry i and j. 
      Default is p=2, or Euclidean distance.
    '''
    weights = embedding_layer.weight.data
    distance_matrix = torch.cdist(weights, weights, p=p)
    return distance_matrix

def sample_closest_words_transformer(index, distance_matrix, top_k, weighted):
    '''
    Given a vocabulary index and a distance matrix, select a replacement from
    the nearest neighbors. If `weighted=True`, sampling probabilities will be 
    scaled proportionally to the returned distances between the original and 
    candidates.
    '''
    dists = distance_matrix[index]
    closest_ids = dists.argsort()[1:(top_k+1)]
    
    if weighted:
        probas = np.array(1 / dists[closest_ids]).astype(float)
        probas /= probas.sum()
    else:
        probas = None
    
    new_token = np.random.choice(closest_ids, p=probas)
    return new_token

def perturb_input_ids_torch(ids, 
                            distance_matrix, 
                            replace_p=DEFAULT_REPLACEMENT_PROB, 
                            top_k=DEFAULT_SAMPLE_TOP_K, 
                            weighted=True):
    '''
    Given a tokenized input (i.e. the `input_ids` returned by a HuggingFace 
    tokenizer) and a corresponding word embedding distance matrix, perform
    random perturbations based on nearest neighbors for a random subset of 
    tokens.
    
    Inputs:
      - ids: a torch.LongTensor of token IDs used to index `distance_matrix`
      - distance_matrix: a N x N distance matrix where N is the size of the 
        vocabulary used in the tokenizer/embedding layer
      - replace_p: the probability that each token will be replaced
      - top_k: the number of nearest neighbors to consider for sampling
      - weighted: if True, sample proportionally to the similarity of the `top_k`
        returned neighbors. otherwise, sample the `top_k` neighbors uniformly
    
    Output: 
      - perturbed: a tensor with the same shape as `ids`, with some token IDs
        replaced through random sampling of nearest neighbors
    '''
    # create a boolean mask of the same shape as `ids`, denoting whether
    # each token is to be perturbed
    replace_mask = torch.rand(*ids.shape) <= replace_p
    replacements = [sample_closest_words_transformer(t, 
                                                     distance_matrix, 
                                                     top_k=top_k, 
                                                     weighted=weighted) for t in ids[replace_mask]]
    
    perturbed = ids.clone()
    perturbed[replace_mask] = torch.tensor(replacements)
    return perturbed

## ----- Gensim-based replacement for BOW models ----- ##
def sample_closest_words_gensim(word, wv, top_k, weighted):
    '''
    Given a single word and a set of word vectors (a KeyedVectors object from 
    gensim), sample a replacement word from its `top_k` nearest neighbors. 
    
    Inputs:
      - word: a word to replace; used to query `wv.most_similar()`
      - wv: a gensim KeyedVectors object (e.g. from Word2Vec.wv or similar)
      - top_k: how many neighbors to consider
      - weighted: if True, sample proportionally to similarity
      
    Output:
      - a single replacement word
    '''
    try:
        words, probas = zip(*wv.most_similar(word, topn=top_k))
    except KeyError:
        # if the word isn't found in the vocab, pick a random word from the vocab 
        # TODO: is this valid?
        return np.random.choice(wv.index_to_key)
    
    probas = np.array(probas)
    probas /= probas.sum()
    
    selection = np.random.choice(words, p=probas if weighted else None)
    return selection

def perturb_input_text_gensim(text, 
                              wv,
                              replace_p=DEFAULT_REPLACEMENT_PROB,
                              top_k=DEFAULT_SAMPLE_TOP_K,
                              weighted=True,
                              token_sep=r"[^a-z]+"):
    '''
    Given an input text and word vectors, tokenize and perturb through sampling 
    as described.
    
    Inputs:
      - text: the original text, normalized in the same way expected by your 
        word vectors
      - wv: a gensim KeyedVectors object (e.g. `gensim.models.Word2Vec.wv` or 
        similar)
      - replace_p: the probability that each token will be replaced
      - top_k: the number of nearest neighbors to consider for sampling
      - weighted: if True, sample proportionally to the similarity of the `top_k`
        returned neighbors. otherwise, sample the `top_k` neighbors uniformly
    '''
    og_tokens = np.array(re.split(token_sep, text))
    replace_mask = np.random.uniform(size=len(og_tokens)) <= replace_p
    
    replacements = [sample_closest_words_gensim(word, wv, top_k=top_k, weighted=weighted) for word in og_tokens[replace_mask]]
    perturbed = og_tokens.copy()
    perturbed[replace_mask] = replacements
    return ' '.join(perturbed)