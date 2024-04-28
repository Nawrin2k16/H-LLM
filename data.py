import torch
import sentencepiece as spm
from datasets import load_dataset
import torch.nn.functional as F



def decode_text(encoded_tensor, sp_model_path):
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor(model_file=f"{sp_model_path}.model")

    # Print encoded token IDs for debugging
    print("Encoded Token IDs:", encoded_tensor)

    # Decode the list of integers to text
    try:
        decoded_text = sp.decode_ids(encoded_tensor)
    except Exception as e:
        print("Error during decoding:", e)
        return None
    
    print(decoded_text)
    return decoded_text


def encode_text(paragraphs, model_path):
    sp = spm.SentencePieceProcessor(model_file=f"{model_path}.model")
    encoded_paragraphs = []
    for paragraph in paragraphs:
        encoded_paragraph = sp.encode_as_pieces(paragraph)
        encoded_paragraphs.append(encoded_paragraph)
    # Join the lists of tokens into single strings
    encoded_paragraphs = [''.join(tokens) for tokens in encoded_paragraphs]


    return encoded_paragraphs



def collect_data(dataset_name, model_path):
    # Load the dataset from Hugging Face's datasets library
    config = 'plain_text'
    dataset = load_dataset(dataset_name, config, split='train[:1%]' )
    
    # Extract articles from the dataset
    articles = dataset['text']

    # Encode articles using SentencePiece model
    encoded_articles = encode_text(articles, model_path)

    # Convert each encoded article to a PyTorch tensor
    tensor_articles = [torch.tensor(encoded_article) for encoded_article in encoded_articles]

    sp = spm.SentencePieceProcessor(model_file=f"{model_path}.model")
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    return tensor_articles, vocab


def get_batches(data, split, batch_size, context_window):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    if split == 'train':
        batch_data = train
    elif split == 'val':
        batch_data = val
    elif split == 'test':
        batch_data = test


    # pick random starting points
    max_ix = max(0, batch_data[0].size(0) - context_window - 1)
    ix = torch.randint(0, max_ix, (batch_size,))
    input_x = torch.stack([batch_data[0][i:i+context_window] for i in ix]).long()
    output_y = torch.stack([batch_data[0][i+1:i+context_window+1] for i in ix]).long()

    return input_x, output_y

    
def generate(model, text, max_new_tokens=100, model_path="/home/nawrin/H_LLM/scratch/saved_models/bpe_model"):
    sp = spm.SentencePieceProcessor(model_file=f"{model_path}/spm.model")
    encoded_text = sp.encode_as_ids(text)
    #print(encoded_text)
    idx = torch.tensor([encoded_text], dtype=torch.long)  
    

    if next(model.parameters()).is_cuda:
        idx = idx.cuda()

    id = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(idx[:, id:-1], idx[:, id+1:]) 

            last_logits = logits[:, -1, :]
            probabilities = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            id += 1
            idx = torch.cat([idx, next_token], dim=-1)

    generated_text = sp.decode_ids(idx.squeeze(0).tolist())
    return generated_text