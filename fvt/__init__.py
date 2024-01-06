import abc
import torch.nn as nn


class AbstractVocabularyTransfer(metaclass=abc.ABCMeta):

    def __init__(self):
        self.tokens_map = None

    @staticmethod
    @abc.abstractmethod
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
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
    def train_tokenizer(data, gen_tokenizer, vocab_size, **kwargs):
        """
        Train an HF tokenizer with the specified vocab size.

        :param data: a list of textual sequences to train the tokenizer with
        :param gen_tokenizer: a general-purpose tokenizer.
        :param vocab_size: int. Vocabulary size for the new trained tokenizer
        :param kwargs: no kwargs

        :return: A new trained tokenizer in the in-domain data
        """
        in_tokenizer = gen_tokenizer.train_new_from_iterator(data, vocab_size)

        return in_tokenizer

    @abc.abstractmethod
    def embeddings_assignment(self, tokens_map, gen_model, **kwargs):
        raise NotImplementedError

    def update_model_embeddings(self, gen_model, in_matrix, **kwargs):
        """
        Method that replaces the embeddings of a given LM with in_matrix.

        :param gen_model: An huggingface model, e.g. bert
        :param in_matrix: (2-d torch.Tensor) The new embedding matrix.
        :param kwargs: no kwargs

        :return: A new LM model with replaced embeddings
        """

        # Change the model's embedding matrix
        gen_model.get_input_embeddings().weight = nn.Parameter(in_matrix)
        gen_model.config.vocab_size = in_matrix.shape[0]

        tie_weights = kwargs.get('tie_weights', True)
        if tie_weights:
            # Tie the model's weights
            gen_model.tie_weights()

        return gen_model

    def transfer(self, in_tokenizer, gen_tokenizer, gen_model, **kwargs):
        """
        Method that returns a new LM model with transferred embeddings.

        :param in_tokenizer: Any huggingface tokenizer
        :param gen_tokenizer: Any huggingface tokenizer
        :param gen_model: Any huggingface model
        :param kwargs: no kwargs

        :return: A new in_domain model
        """

        self.tokens_map = self.tokens_mapping(in_tokenizer, gen_tokenizer)
        in_matrix = self.embeddings_assignment(self.tokens_map, gen_model)
        in_model = self.update_model_embeddings(gen_model, in_matrix)

        return in_model
