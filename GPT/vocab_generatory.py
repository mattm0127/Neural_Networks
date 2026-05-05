from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def tokenizer_train():
    text_path = r"LSTM\training_data\tiny_shakespear.txt"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=1024)
    tokenizer.train(files=[text_path], trainer=trainer)

    tokenizer.save('GPT/shakes_bpe.json')

if __name__ == "__main__":
    tokenizer_train()