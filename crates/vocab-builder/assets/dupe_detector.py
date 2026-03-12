#!/usr/bin/env python3
"""
Script to find duplicate words in a text file (one word per line).
Ignores case when detecting duplicates but prints words with original case.
"""

from collections import defaultdict
import sys


def find_duplicates(filename):
    """
    Find duplicate words in a file, ignoring case.

    Args:
        filename: Path to the text file to scan

    Returns:
        Dictionary mapping lowercase words to list of original case variations
    """
    word_occurrences = defaultdict(list)

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                word = line.strip()
                if word:  # Skip empty lines
                    word_occurrences[word.lower()].append((word, line_num))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return word_occurrences


def print_duplicates(word_occurrences):
    """Print all duplicate words with their case variations."""
    duplicates_found = False

    for lowercase_word, occurrences in sorted(word_occurrences.items()):
        if len(occurrences) > 1:
            duplicates_found = True
            print(f"\nDuplicate word (appears {len(occurrences)} times):")
            for word, line_num in occurrences:
                print(f"  Line {line_num}: {word}")

    if not duplicates_found:
        print("No duplicates found.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python find_duplicates.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    word_occurrences = find_duplicates(filename)
    print_duplicates(word_occurrences)


if __name__ == "__main__":
    main()