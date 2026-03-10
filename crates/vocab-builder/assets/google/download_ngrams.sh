#!/bin/bash
# download_ngrams.sh - Download and aggregate Google Books Ngrams

set -e  # Exit on error

# Configuration
DOWNLOAD_DIR="./ngrams_download"
OUTPUT_DIR="./ngrams_processed"
UNIGRAM_OUTPUT="$OUTPUT_DIR/all_1grams.txt"
BIGRAM_OUTPUT="$OUTPUT_DIR/all_2grams.txt"

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$OUTPUT_DIR"

# Base URL
BASE_URL="http://storage.googleapis.com/books/ngrams/books"

echo "=== Google Books Ngrams Downloader ==="
echo "This will download ~2.6GB for 1-grams and ~30GB for 2-grams"
echo "Download directory: $DOWNLOAD_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to download and process 1-grams
download_unigrams() {
    echo "=== Downloading 1-grams ==="

    # Clear output file if it exists
    > "$UNIGRAM_OUTPUT"

    # Letters a-z for 1-grams
    for letter in {a..z}; do
        filename="googlebooks-eng-all-1gram-20120701-${letter}.gz"
        url="${BASE_URL}/${filename}"

        echo "Downloading $filename..."
        wget -q -O "$DOWNLOAD_DIR/$filename" "$url" || {
            echo "Failed to download $filename, skipping..."
            continue
        }

        echo "Processing $filename..."
        gunzip -c "$DOWNLOAD_DIR/$filename" >> "$UNIGRAM_OUTPUT"

        # Remove downloaded file to save space
        rm "$DOWNLOAD_DIR/$filename"

        echo "Completed $letter"
    done

    echo "=== 1-grams complete! ==="
    echo "Output: $UNIGRAM_OUTPUT"
    echo "Lines: $(wc -l < "$UNIGRAM_OUTPUT")"
}

# Function to download and process 2-grams
download_bigrams() {
    echo "=== Downloading 2-grams ==="
    echo "WARNING: This will download ~30GB and take a long time!"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping 2-grams"
        return
    fi

    # Clear output file if it exists
    > "$BIGRAM_OUTPUT"

    # Two-letter combinations for 2-grams (aa, ab, ..., zz)
    for letter1 in {a..z}; do
        for letter2 in {a..z}; do
            filename="googlebooks-eng-all-2gram-20120701-${letter1}${letter2}.gz"
            url="${BASE_URL}/${filename}"

            echo "Downloading $filename..."
            wget -q -O "$DOWNLOAD_DIR/$filename" "$url" || {
                echo "Failed to download $filename, skipping..."
                continue
            }

            echo "Processing $filename..."
            gunzip -c "$DOWNLOAD_DIR/$filename" >> "$BIGRAM_OUTPUT"

            # Remove downloaded file to save space
            rm "$DOWNLOAD_DIR/$filename"

            echo "Completed ${letter1}${letter2}"
        done
    done

    echo "=== 2-grams complete! ==="
    echo "Output: $BIGRAM_OUTPUT"
    echo "Lines: $(wc -l < "$BIGRAM_OUTPUT")"
}

# Main execution
echo "What would you like to download?"
echo "1) 1-grams only (~2.6GB)"
echo "2) 2-grams only (~30GB)"
echo "3) Both 1-grams and 2-grams (~32GB)"
read -p "Choice (1/2/3): " choice

case $choice in
    1)
        download_unigrams
        ;;
    2)
        download_bigrams
        ;;
    3)
        download_unigrams
        download_bigrams
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=== All downloads complete! ==="
echo "You can now process these files with the aggregation script."