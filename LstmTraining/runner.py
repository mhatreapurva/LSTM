from train import LstmTrainer
from tokenizer import LstmTokenizer

if __name__ == "__main__":
    tokenizer = LstmTokenizer()
    trainer = LstmTrainer(tokenizer_obj=tokenizer)
    trainer.train_lstm()
