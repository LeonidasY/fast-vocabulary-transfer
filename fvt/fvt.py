import re
import torch

from fvt import VocabularyTransfer


class FastVocabularyTransfer(VocabularyTransfer):

    def __init__(self):
        super(FastVocabularyTransfer, self).__init__()

    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        gen_vocab = gen_tokenizer.get_vocab()
        in_vocab = in_tokenizer.get_vocab()

        tokens_map = {}
        for new_token, new_index in list(in_vocab.items()):
            # If the same token exists in the old vocabulary, take its embedding
            if new_token in gen_vocab:
                old_index = gen_vocab[new_token]
                tokens_map[in_vocab[new_token]] = [old_index]
            else:
                # if not, tokenise the new token using the old vocabulary
                # Remove '##' from the beginning of the subtoken
                new_token = re.sub("^(##|Ġ)", '', new_token)
                token_partition = gen_tokenizer.tokenize(new_token)
                tokens_map[in_vocab[new_token]] = [gen_vocab[old_token] for old_token in token_partition]

        return tokens_map

    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        gen_matrix = gen_model.get_input_embeddings().weight
        in_matrix = torch.zeros(len(tokens_map), gen_matrix.shape[1])

        for new_index, old_token_indexes in tokens_map.items():
            old_embedding = torch.mean(gen_matrix[old_token_indexes], axis=0)
            in_matrix[new_index] = old_embedding

        return in_matrix

        # for new_token, new_index in list(in_vocab.items()):
        #
        #     # If the same token exists in the old vocabulary, take its embedding
        #     if new_token in gen_vocab:
        #
        #         old_index = gen_vocab[new_token]
        #         in_matrix[new_index] = gen_matrix[old_index]
        #
        #     else:
        #         # if not, tokenise the new token using the old vocabulary
        #         # Remove '##' from the beginning of the subtoken
        #         # new_token = re.sub("^(##|Ġ)", '', new_token)
        #         # partition = self.gen_tokenizer.tokenize(new_token)
        #
        #         new_embedding = []
        #         for old_token in partition:
        #             old_index = gen_vocab[old_token]
        #             old_embedding = gen_matrix[old_index]
        #             new_embedding.append(old_embedding)
        #
        #         # Initialise the new embedding as the average of its old embeddings
        #         new_embedding = torch.vstack(new_embedding)
        #         new_embedding = torch.mean(new_embedding, 0)
        #         in_matrix[new_index] = new_embedding