import csv
import random
import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

import pickle

def parse_dataset() -> dict[str, list[str]]:
    """
    Parse the dataset from a CSV file and separate positive and negative examples.

    Returns:
        dict: A dictionary with "Positive" and "Negative" keys containing lists of examples.
    """
    reader = csv.reader(open("DATASET.csv"))

    data = {"Positive": [], "Negative": []}

    for line in reader:
        # Skip the first line
        if reader.line_num == 1:
            print("Headers", line)
            print()
            continue

        # Print the line
        if line[0] == "LITERAL":
            data["Negative"].append(line)
        
        else:
            data["Positive"].append(line)
    
    return data

# Stack examples
# Create a dataset of pairs of positive and negative examples (x_pos, x_neg)
def generate_example_pairs(data, num_pairs=1000):
    """
    Generate a list of tuples containing positive and negative examples without repetition.

    Args:
        data (dict): A dictionary with "Positive" and "Negative" keys containing lists of examples.
        num_pairs (int): The number of pairs to generate.

    Returns:
        list: A list of tuples (x_pos, x_neg), where x_pos is a positive example and x_neg is a negative example.
    """
    if num_pairs > min(len(data["Positive"]), len(data["Negative"])):
        raise ValueError("Not enough unique examples to generate the requested number of pairs.")

    positive_samples = random.sample(data["Positive"], num_pairs)
    negative_samples = random.sample(data["Negative"], num_pairs)

    # Extract only the 4th object (index 3) from each sample
    pairs = [(x_pos[3], x_neg[3]) for x_pos, x_neg in zip(positive_samples, negative_samples)]
    return pairs

def extract_embeddings(model, tokenizer, text):
    """
    Extract embeddings for a given text using the specified model and tokenizer.

    Args:
        model: The pre-trained transformer model.
        tokenizer: The tokenizer corresponding to the model.
        text (str): The input text to extract embeddings for.

    Returns:
        torch.Tensor: The embedding vector for the input text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token) as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.squeeze(0)

def generate_embeddings(num_pairs=1000):
    data = parse_dataset()
    print(f"Parsed {len(data['Positive'])} positive and {len(data['Negative'])} negative examples.")

    # Load the LLaMA-90B model and tokenizer
    print("Loading LLaMA-90B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    model = AutoModel.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map="auto")
    model.eval()

    # Generate example pairs
    try:
        example_pairs = generate_example_pairs(data, num_pairs)
        print(f"Generated {len(example_pairs)} example pairs.")
    except ValueError as e:
        print(e)
        exit()

    # Extract embeddings for each pair
    embeddings = []
    for x_pos, x_neg in tqdm.tqdm(example_pairs):
        x_pos_embed = extract_embeddings(model, tokenizer, x_pos)
        x_neg_embed = extract_embeddings(model, tokenizer, x_neg)
        embeddings.append((x_pos_embed.tolist(), x_neg_embed.tolist()))

    # Save the embeddings to a file
    with open("example_pairs_embeddings.txt", "w") as f:
        for x_pos_embed, x_neg_embed in embeddings:
            f.write(f"{x_pos_embed}  --  {x_neg_embed}\n")

    print("Embeddings saved to 'example_pairs_embeddings.txt'.")

def parse_embeddings(fname="example_pairs_embeddings.txt"):
    with open(fname) as f:
        positive_embeddings = []
        negative_embeddings = []
        for line in f:
            x_pos_embed, x_neg_embed = line.strip().split("  --  ")
            x_pos_embed = torch.tensor(eval(x_pos_embed))
            x_neg_embed = torch.tensor(eval(x_neg_embed))
            positive_embeddings.append(x_pos_embed)
            negative_embeddings.append(x_neg_embed)
    
    return torch.vstack(positive_embeddings), torch.stack(negative_embeddings)

def load_embeddings():
    """
    Load the embeddings from the file and return them as tensors.

    Returns:
        tuple: A tuple containing two tensors: positive and negative embeddings.
    """
    try:
        pos_embeddings, neg_embeddings = pickle.load(open("embeddings.pkl", "rb"))
    except FileNotFoundError:
        print("Embeddings file not found. Generating new embeddings...")
        generate_embeddings()
        pickle.dump((pos_embeddings, neg_embeddings), open("embeddings.pkl", "wb"))
        print("Embeddings loaded and saved to 'embeddings.pkl'.")
    
    return pos_embeddings, neg_embeddings

if __name__ == "__main__":
    pos_embeddings, neg_embeddings = load_embeddings()
    print(pos_embeddings.shape, neg_embeddings.shape)