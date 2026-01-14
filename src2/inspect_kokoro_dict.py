#!/usr/bin/env python3
"""Inspect Kokoro phoneme dictionary."""

import pickle
from pathlib import Path

# Load the dictionary
dict_path = Path("models/TTS/lang_phoneme_dict.pkl")

with open(dict_path, "rb") as f:
    phoneme_dict = pickle.load(f)

print(f"Type: {type(phoneme_dict)}")
print(f"Keys: {list(phoneme_dict.keys()) if isinstance(phoneme_dict, dict) else 'Not a dict'}")

if isinstance(phoneme_dict, dict):
    # Sample some entries
    for i, (key, value) in enumerate(list(phoneme_dict.items())[:20]):
        print(f"{key}: {value}")
        if i >= 19:
            break
    print(f"\nTotal entries: {len(phoneme_dict)}")

    # Check for specific characters
    test_chars = ['h', 'e', 'l', 'o', ' ', 'H', 'E', 'L', 'O']
    print("\nTest characters:")
    for char in test_chars:
        if char in phoneme_dict:
            print(f"  '{char}': {phoneme_dict[char]}")
