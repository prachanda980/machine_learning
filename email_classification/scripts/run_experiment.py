# Example script to run training and prediction
from src.train_model import train
from src.predict import predict_email

if __name__ == "__main__":
    train()
    print(predict_email("Congratulations, you have won a prize!"))
