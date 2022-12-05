import abc

import torch.nn as nn


class AbstractVocabularyTransfer(metaclass=abc.ABCMeta):
    def __init__(self):

        self.tokens_map = {}
        self.in_matrix = None
        self.in_tokenizer = None
        self.in_model = None

    @staticmethod
    @abc.abstractmethod
    def train_tokenizer(in_domain_data, gen_tokenizer, vocab_size, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def tokens_mapping(self, in_tokenizer, gen_tokenizer, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_embeddings(self, gen_model, in_vocab, in_matrix, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def transfer(self, in_domain_data, gen_tokenizer, gen_model, vocab_size, **kwargs):
        raise NotImplementedError


class VocabularyTransfer(AbstractVocabularyTransfer):
    def __init__(self):
        super(VocabularyTransfer, self).__init__()

    @staticmethod
    def train_tokenizer(in_domain_data, gen_tokenizer, vocab_size, **kwargs):
        in_tokenizer = gen_tokenizer.train_new_from_iterator(in_domain_data, vocab_size)

        return in_tokenizer

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        raise NotImplementedError

    def update_model_embeddings(self, gen_model, in_vocab, in_matrix, **kwargs):
        # Change the model's embedding matrix
        self.in_model = gen_model
        self.in_model.get_input_embeddings().weight = nn.Parameter(in_matrix)
        self.in_model.config.vocab_size = len(in_vocab)

        tie_weights = kwargs.get("tie_weights", True)
        if tie_weights:
            # Tie the model's weights
            self.in_model.tie_weights()

        return self.in_model

    def transfer(self, in_domain_data, gen_tokenizer, gen_model, vocab_size, **kwargs):
        self.in_tokenizer = self.train_tokenizer(in_domain_data, gen_tokenizer, vocab_size, **kwargs)
        self.tokens_map = self.tokens_mapping(self.in_tokenizer, gen_tokenizer)
        self.in_matrix = self.embeddings_assignment(self.tokens_map, gen_model)
        self.in_model = self.update_model_embeddings(gen_model, self.in_tokenizer.get_vocab(), self.in_matrix)

        return self.in_tokenizer, self.in_model
