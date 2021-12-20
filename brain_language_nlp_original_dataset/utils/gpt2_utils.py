import torch
import numpy as np
from transformers import *
import time as tm


def get_gpt2_layer_representations(seq_len, text_array, remove_chars, word_ind_to_extract):
    ModelClass = GPT2Model
    ModelConfig = GPT2Config
    ModelTokenizer = GPT2Tokenizer
    pretrained_weight_shortcuts = 'gpt2'
    config = ModelConfig.from_pretrained(pretrained_weight_shortcuts, output_hidden_states=True)
    model = ModelClass.from_pretrained(pretrained_weight_shortcuts, config=config)
    tokenizer = ModelTokenizer.from_pretrained(pretrained_weight_shortcuts)
    model.eval()

    # get the token embeddings
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_model_token_embeddings([word], tokenizer, model, remove_chars, ModelClass)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))

    # where to store layer-wise bert embeddings of particular length
    MODEL_DICT = {}
    iter_length = 0
    if pretrained_weight_shortcuts == 'distilgpt2' or pretrained_weight_shortcuts == 'distilbert-base-cased':
        iter_length = 6
    else:
        iter_length = 12

    for layer in range(iter_length):
        MODEL_DICT[layer] = []
    MODEL_DICT[-1] = token_embeddings

    if word_ind_to_extract < 0:
        # the index is specified from the end of the array, so invert the index
        if ModelClass == BertModel or ModelClass == DistilBertModel or ModelClass == GPT2Model:
            from_start_word_ind_to_extract = seq_len + 2 + word_ind_to_extract  # add 2 for CLS + SEP tokens
        else:
            from_start_word_ind_to_extract = seq_len + word_ind_to_extract
    else:
        from_start_word_ind_to_extract = word_ind_to_extract

    start_time = tm.time()

    # before we've seen enough words to make up the sequence length, add the representation for the last word 'seq_len' times
    word_seq = text_array[:seq_len]
    for _ in range(seq_len):
        MODEL_DICT = add_avrg_token_embedding_for_specific_word(word_seq,
                                                                tokenizer,
                                                                model,
                                                                remove_chars,
                                                                from_start_word_ind_to_extract,
                                                                MODEL_DICT, ModelClass)

    # then add the embedding of the last word in a sequence as the embedding for the sequence
    for end_curr_seq in range(seq_len, len(text_array)):
        word_seq = text_array[end_curr_seq - seq_len + 1:end_curr_seq + 1]
        MODEL_DICT = add_avrg_token_embedding_for_specific_word(word_seq,
                                                                tokenizer,
                                                                model,
                                                                remove_chars,
                                                                from_start_word_ind_to_extract,
                                                                MODEL_DICT, ModelClass)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time() - start_time))
            start_time = tm.time()

    print('Done extracting sequences of length {}'.format(seq_len))

    return MODEL_DICT


# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
def predict_model_embeddings(words_in_array, tokenizer, model, remove_chars, ModelClass):
    for word in words_in_array:
        if word in remove_chars:
            print(
                'An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1

    n_seq_tokens = 0
    seq_tokens = []

    word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    for i, word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []  # initialize token indices array for current word

        if word in ['[CLS]', '[SEP]']:  # [CLS] and [SEP] are already tokenized
            word_tokens = [word]
        else:
            word_tokens = tokenizer.tokenize(word)

        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1

    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    encoded_layers = model(tokens_tensor)
    if ModelClass == BertModel or ModelClass == GPT2Model:
        hidden_states = list(encoded_layers[2][1:])
    if ModelClass == DistilBertModel:
        hidden_states = list(encoded_layers[1][1:])

    return hidden_states, word_ind_to_token_ind


# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_model_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        model_dict[specific_layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            model_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :],
                                             0))  # avrg over all tokens for specified word
    return model_dict


# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: BERT tokenizer
# model: BERT model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# bert_dict: where to save the extracted embeddings
def add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, remove_chars, from_start_word_ind_to_extract,
                                               model_dict, ModelClass):
    # if ModelClass == BertModel or ModelClass == DistilBertModel:
    word_seq = ['[CLS]'] + list(word_seq) + ['[SEP]']
    all_sequence_embeddings, word_ind_to_token_ind = predict_model_embeddings(word_seq, tokenizer, model, remove_chars,
                                                                              ModelClass)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_model_embedding(model_dict, all_sequence_embeddings, token_inds_to_avrg)

    return model_dict


# get the MODEL token embeddings
def get_model_token_embeddings(words_in_array, tokenizer, model, remove_chars, ModelClass):
    for word in words_in_array:
        if word in remove_chars:
            print(
                'An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1

    n_seq_tokens = 0
    seq_tokens = []

    word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens

    for i, word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []  # initialize token indices array for current word

        if word in ['[CLS]', '[SEP]']:  # [CLS] and [SEP] are already tokenized
            word_tokens = [word]
        else:
            word_tokens = tokenizer.tokenize(word)

        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1

    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)

    # convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    if ModelClass == BertModel or ModelClass == DistilBertModel:
        token_embeddings = model.embeddings.forward(tokens_tensor)
    else:
        token_embeddings = model.wte.forward(tokens_tensor)

    return token_embeddings



