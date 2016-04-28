Introduction:

How to do deepwalk processing ?
usage: deepwalk [-h] [--debug] [--format FORMAT] --input [INPUT] [-l LOG]
                [--matfile-variable-name MATFILE_VARIABLE_NAME]
                [--max-memory-data-size MAX_MEMORY_DATA_SIZE]
                [--number-walks NUMBER_WALKS] --output OUTPUT
                [--representation-size REPRESENTATION_SIZE] [--seed SEED]
                [--undirected UNDIRECTED] [--vertex-freq-degree]
                [--walk-length WALK_LENGTH] [--window-size WINDOW_SIZE]
                [--workers WORKERS] [--mode MODE]

EXAMPLE :
deepwalk \
  --input cleaned.adjlist \
  --output out_new.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 4 \
  --representation-size 64 \
  --format edgelist


Experiment :
deepwalk \
  --input cleaned.adjlist \
  --output out_new.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 4 \
  --representation-size 64 \
  --format edgelist
deepwalk \
  --input cleaned.adjlist \
  --output out_new.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 4 \
  --representation-size 64 \
  --format edgelist




How to do dimension reduction ?

python dim_reduction.py

TODO:
1. Make ReadInfo Class more perfect :
    1. Adding get voc size and lyrics number functions
    2. Adding functions that can get the title, artist , an album of the particular lyricsinfo
    3. Adding functions that can get the title, artist an album of all lyrics in an array


NOTE:
1.
In filename containing, "cleaned" related to the ones that are after the cleaning pre-processing in adjlist generation phases.
I used lemmatization and better tokenization in the ReadInfo.py initialization function.
