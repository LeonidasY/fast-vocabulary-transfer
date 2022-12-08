from transformers import AutoTokenizer, AutoModelForTokenClassification
from fvt.fvt import FastVocabularyTransfer

if __name__ == "__main__":

    pretrained_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pretrained_model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")

    # load your dataset here
    in_domain_data = ['A list of strings', '...']

    fvt = FastVocabularyTransfer()
    in_tokenizer, in_model = fvt.transfer(
        in_domain_data=in_domain_data,
        gen_tokenizer=pretrained_tokenizer,
        gen_model=pretrained_model,
        vocab_size=10000
    )

