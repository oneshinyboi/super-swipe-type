#!/usr/bin/env python3
"""
Script to remove capitalized and full caps duplicate words from a text file.
When a word appears in multiple cases, keeps only the lowercase version.
Ignores apostrophes when comparing words.
"""

import sys


def normalize_for_comparison(word):
    """Remove apostrophes and convert to lowercase for comparison."""
    return word.replace("'", "").lower()


def is_title_case(word):
    """Check if word is title case (first char upper, rest lower)."""
    return word and word[0].isupper() and word[1:].islower()


def is_all_caps(word):
    """Check if word is all uppercase (ignoring apostrophes)."""
    letters_only = ''.join(c for c in word if c.isalpha())
    return letters_only and letters_only.isupper()


def process_file(input_filename, output_filename=None):
    """
    Remove capitalized and full caps duplicates from word list.

    Args:
        input_filename: Path to input file
        output_filename: Path to output file (optional, defaults to input_file.cleaned.txt)
    """
    if output_filename is None:
        output_filename = input_filename.rsplit('.', 1)[0] + '.cleaned.txt'

    # Read all words
    words = []
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            words = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Build set of normalized lowercase versions
    normalized_lowercase = set()
    for word in words:
        normalized = normalize_for_comparison(word)
        # Check if this word is actually lowercase
        if word.lower() == word:
            normalized_lowercase.add(normalized)

    # Filter out capitalized and full caps duplicates
    filtered_words = []
    removed_count = 0

    for word in words:
        normalized = normalize_for_comparison(word)
        should_remove = False

        # Check if lowercase version exists
        if normalized in normalized_lowercase:
            # Remove if title case (e.g., "Yuletide" when "yuletide" exists)
            if is_title_case(word) and word.lower() != word:
                print(f"Removing title case: {word} (lowercase version exists)")
                should_remove = True
            # Remove if all caps (e.g., "HELLO" when "hello" exists)
            elif is_all_caps(word) and word.lower() != word:
                print(f"Removing all caps: {word} (lowercase version exists)")
                should_remove = True

        if should_remove:
            removed_count += 1
        else:
            filtered_words.append(word)

    # Write cleaned file
    try:
        with open(output_filename, 'w', encoding='utf-8') as file:
            for word in filtered_words:
                file.write(word + '\n')

        print(f"\nProcessing complete:")
        print(f"  Original words: {len(words)}")
        print(f"  Removed: {removed_count}")
        print(f"  Remaining: {len(filtered_words)}")
        print(f"  Output file: {output_filename}")

    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_capitalized_duplicates.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    process_file(input_file, output_file)


if __name__ == "__main__":
    main()