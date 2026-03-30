#!/bin/bash

# Extraction script for all olmOCR subsets
DEST="/path/to/olmocr-dataset"  # Where to extract olmOCR training data
DATASET="allenai/olmOCR-mix-1025"

echo "Starting olmOCR extraction to: $DEST"
echo "================================================================"

# Array of subsets
SUBSETS=("00_documents" "01_books" "02_loc_transcripts" "03_national_archives")

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Extracting subset: $subset"
    echo "================================================================"
    
    echo "Processing train split..."
    python -m olmocr.data.prepare_olmocrmix \
        --dataset-path $DATASET \
        --destination $DEST \
        --subset $subset \
        --split train
    
    echo "Processing eval split..."
    python -m olmocr.data.prepare_olmocrmix \
        --dataset-path $DATASET \
        --destination $DEST \
        --subset $subset \
        --split eval
    
    echo "✓ Completed $subset"
done

echo ""
echo "================================================================"
echo "All extractions complete!"
echo "Data location: $DEST"
echo "================================================================"
