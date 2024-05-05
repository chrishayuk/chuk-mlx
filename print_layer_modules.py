import argparse
from utils.huggingface_utils import load_from_hub
from transformers import AutoModel

def print_model_layer_modules(model):
    # loop through the modules
    for name, _ in model.named_modules():
        # print the name of the 
        print(name)
    
def main(model_name):
    # load the model from huggingface
    model_path = load_from_hub(args.model)
    model = AutoModel.from_pretrained(model_path)

    # Print the model layers
    print_model_layer_modules(model)

if __name__ == "__main__":
    # setup the parser
    parser = argparse.ArgumentParser(description='print layer modules')

    # parse model name and config path
    parser.add_argument("--model", required=False, default="meta-llama/Meta-Llama-3-8b-Instruct", help='model e.g. meta-llama/Llama-3-8b-Instruct')

    # parse the arguments
    args = parser.parse_args()

    # execute
    main(args.model)
