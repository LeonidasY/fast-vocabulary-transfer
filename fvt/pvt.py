import torch

from fvt import VocabularyTransfer


class PartialVocabularyTransfer(VocabularyTransfer):

    def __init__(self, seed=0):
        super(PartialVocabularyTransfer, self).__init__()
        self.seed = seed

    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        gen_vocab = gen_tokenizer.get_vocab()
        in_vocab = in_tokenizer.get_vocab()

        tokens_map = {}
        for new_token, new_index in in_vocab.items():
            if new_token in gen_vocab:
                # if the same token exists in the old vocabulary, take its embedding
                old_index = gen_vocab[new_token]
                tokens_map[new_index] = [old_index]
            
            else:
                tokens_map[new_index] = []  # no index to map to

        return tokens_map

    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        gen_matrix = gen_model.get_input_embeddings().weight
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1])

        for new_index, old_indices in tokens_map.items():
            if old_indices and len(old_indices) == 1:
                old_index = old_indices[0]
                in_matrix[new_index] = gen_matrix[old_index]
            
            elif len(old_indices) > 1:
                raise AttributeError(
                    f'PVT does not support 1-n mappings, multiple elements in old_token_indexes found: {old_indices}'
                )
            
            else:
                # if not, initialise a random vector for the new token
                torch.manual_seed(self.seed)
                in_matrix[new_index] = torch.rand(1, gen_matrix.shape[1])

        return in_matrix
