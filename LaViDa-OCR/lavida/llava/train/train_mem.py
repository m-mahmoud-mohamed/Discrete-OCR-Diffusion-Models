from llava.train.train import train
import torch._dynamo
import warnings

torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=FutureWarning) 

if __name__ == "__main__":
    train()
