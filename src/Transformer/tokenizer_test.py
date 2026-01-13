from tokenizer import Tokenizer 
from main import VOCAB_SIZE

def test_tokenizer_roundtrip():
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load("models/tokenizer.json")
    
    test_sentence = "This is a test on the BPE tokenizer correctness."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    
    assert test_sentence == decoded
    assert encoded[0] == tokenizer.BOS
    assert encoded[-1] == tokenizer.EOS

def test_empty_string():
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load("models/tokenizer.json")
    assert tokenizer.encode("") == []