import torch
import pandas as pd
import sys
sys.path.append('..')

from tokenizer import Tokenizer
from encoder import Encoder
from decoder import Decoder
from typing import Tuple

VOCAB_SIZE = 8000
EMBEDDING_DIMS = 256
HIDDEN_DIMS = 512
ATTENTION_DIMS = 256

def load_model() -> Tuple[Tokenizer, Encoder, Decoder]:
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load('weights/tokenizer_v1.pt')    

    # Load encoder & decoder
    encoder = Encoder(vocab_size=VOCAB_SIZE + 2, embedding_dim=EMBEDDING_DIMS, hidden_dim=HIDDEN_DIMS)
    decoder = Decoder(vocab_size=VOCAB_SIZE + 2, embedding_dim=EMBEDDING_DIMS, hidden_dim=HIDDEN_DIMS, attention_dim=ATTENTION_DIMS)
    encoder.load_weights('weights/encoder_v3.pt')
    decoder.load_weights('weights/decoder_v3.pt')
    
    print("Ready!\n")
    return tokenizer, encoder, decoder

def translate(text: str, tokenizer: Tokenizer, encoder: Encoder, decoder: Decoder) -> str:
    tokens = tokenizer.encode(text)
    encoder_outputs, hidden, cell = encoder.encode(tokens)
    output = decoder.generate(hidden, cell, encoder_outputs, tokenizer.SOS, tokenizer.EOS, max_length=50)
    return tokenizer.decode(output)

def main():
    print("Seq2Seq Translator (English → Spanish)")
    tokenizer, encoder, decoder = load_model()
    
    print("Type English text to translate. Type 'quit' or 'q' to exit.\n")
    
    while True:
        text = input("English: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n¡Adiós!\n")
            break
        
        if not text:
            continue
        
        spanish = translate(text, tokenizer, encoder, decoder)
        print(f"Spanish: {spanish}\n")

if __name__ == "__main__":
    main()
