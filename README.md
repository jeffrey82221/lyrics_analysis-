**Introduction:

How to do deepwalk processing ?
usage: deepwalk [-h] [--debug] [--format FORMAT] --input [INPUT] [-l LOG]
                [--matfile-variable-name MATFILE_VARIABLE_NAME]
                [--max-memory-data-size MAX_MEMORY_DATA_SIZE]
                [--number-walks NUMBER_WALKS] --output OUTPUT
                [--representation-size REPRESENTATION_SIZE] [--seed SEED]
                [--undirected UNDIRECTED] [--vertex-freq-degree]
                [--walk-length WALK_LENGTH] [--window-size WINDOW_SIZE]
                [--workers WORKERS] [--mode MODE]
***
* EXPERIMENT 1.  Comparing the embedding of different walk length

*DEEPWALK:

deepwalk \
  --input cleaned.adjlist \
  --output w1000.l2.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 2 \
  --representation-size 128 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output w1000.l3.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 3 \
  --representation-size 128 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output w1000.l4.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 4 \
  --representation-size 128 \
  --format edgelist


deepwalk \
  --input cleaned.adjlist \
  --output w1000.l8.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 8 \
  --representation-size 128 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output w1000.l9.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 9 \
  --representation-size 128 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output w1000.l16.d128.embeddings \
  --number-walks 1000 \
  --workers 12 \
  --walk-length 16 \
  --representation-size 128 \
  --format edgelist  

**
*DIMENSION REDUCTION

python dim_reduction.py w1000.l2.d128.embeddings 2
python dim_reduction.py w1000.l3.d128.embeddings 2
python dim_reduction.py w1000.l4.d128.embeddings 2
python dim_reduction.py w1000.l8.d128.embeddings 2
python dim_reduction.py w1000.l9.d128.embeddings 2
python dim_reduction.py w1000.l16.d128.embeddings 2


python dim_reduction.py w1000.l2.d128.embeddings 3
python dim_reduction.py w1000.l3.d128.embeddings 3
python dim_reduction.py w1000.l4.d128.embeddings 3
python dim_reduction.py w1000.l8.d128.embeddings 3
python dim_reduction.py w1000.l9.d128.embeddings 3
python dim_reduction.py w1000.l16.d128.embeddings 3
***
*GENERATE TABLE

python generate_table.py w1000.l2.d128.embeddings.2d
python generate_table.py w1000.l3.d128.embeddings.2d
python generate_table.py w1000.l4.d128.embeddings.2d
python generate_table.py w1000.l8.d128.embeddings.2d
python generate_table.py w1000.l9.d128.embeddings.2d
python generate_table.py w1000.l16.d128.embeddings.2d

python generate_table.py w1000.l2.d128.embeddings.3d
python generate_table.py w1000.l3.d128.embeddings.3d
python generate_table.py w1000.l4.d128.embeddings.3d
python generate_table.py w1000.l8.d128.embeddings.3d
python generate_table.py w1000.l9.d128.embeddings.3d
python generate_table.py w1000.l16.d128.embeddings.3d
**
**RESULT :
1. if the walk length is high enough, the embeddings have similar distribution (higher correlation)
2. if the walk length is low, the embeddings can have very different distribution then other embedding of different walk length
3. note that despite that the walk-length is large, the skip-window will be constraint to about 5, therefore, the distribution with walk length > 5 should be similar.
***
* EXPERIMENT 2. comparing embeddings with different window size

*DEEPWALK

deepwalk \
  --input cleaned.adjlist \
  --output win1.w100.l32.d128.embeddings \
  --number-walks 100 \
  --workers 12 \
  --walk-length 32 \
  --window-size 1 \
  --representation-size 128 \
  --max-memory-data-size 2000000000 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output win2.w100.l32.d128.embeddings \
  --number-walks 100 \
  --workers 12 \
  --walk-length 32 \
  --window-size 2 \
  --representation-size 128 \
  --max-memory-data-size 2000000000 \
  --format edgelist

deepwalk \
  --input cleaned.adjlist \
  --output win3.w100.l32.d128.embeddings \
  --number-walks 100 \
  --workers 12 \
  --walk-length 32 \
  --window-size 3 \
  --representation-size 128 \
  --max-memory-data-size 2000000000 \
  --format edgelist

deepwalk \
--input cleaned.adjlist \
--output win4.w100.l32.d128.embeddings \
--number-walks 100 \
--workers 12 \
--walk-length 32 \
--window-size 4 \
--representation-size 128 \
--max-memory-data-size 2000000000 \
--format edgelist

deepwalk \
--input cleaned.adjlist \
--output win5.w100.l32.d128.embeddings \
--number-walks 100 \
--workers 12 \
--walk-length 32 \
--window-size 5 \
--representation-size 128 \
--max-memory-data-size 2000000000 \
--format edgelist

deepwalk \
--input cleaned.adjlist \
--output win6.w100.l32.d128.embeddings \
--number-walks 100 \
--workers 12 \
--walk-length 32 \
--window-size 6 \
--representation-size 128 \
--max-memory-data-size 2000000000 \
--format edgelist

deepwalk \
--input cleaned.adjlist \
--output win7.w100.l32.d128.embeddings \
--number-walks 100 \
--workers 12 \
--walk-length 32 \
--window-size 7 \
--representation-size 128 \
--max-memory-data-size 2000000000 \
--format edgelist
**
*DIMENSION REDUCTION

python dim_reduction.py win1.w100.l32.d128.embeddings 2
python dim_reduction.py win2.w100.l32.d128.embeddings 2
python dim_reduction.py win3.w100.l32.d128.embeddings 2
python dim_reduction.py win4.w100.l32.d128.embeddings 2
python dim_reduction.py win5.w100.l32.d128.embeddings 2
python dim_reduction.py win6.w100.l32.d128.embeddings 2
python dim_reduction.py win7.w100.l32.d128.embeddings 2
**
*GENERATE TABLE

python generate_table.py win1.w100.l32.d128.embeddings.2d
python generate_table.py win2.w100.l32.d128.embeddings.2d
python generate_table.py win3.w100.l32.d128.embeddings.2d
python generate_table.py win4.w100.l32.d128.embeddings.2d
python generate_table.py win5.w100.l32.d128.embeddings.2d
python generate_table.py win6.w100.l32.d128.embeddings.2d
python generate_table.py win7.w100.l32.d128.embeddings.2d
**

DIMENSION REDUCTION FOR CF SONG EMBEDDING:
python generate_cf_song_table.py CF data/song-embedding-64.csv 2
python generate_cf_song_table.py CF data/song-embedding-128.csv 2

TODO:
1. Make ReadInfo Class more perfect :
    1. Adding get voc size and lyrics number functions
    2. Adding functions that can get the title, artist , an album of the particular lyricsinfo
    3. Adding functions that can get the title, artist an album of all lyrics in an array


**NOTE:
1.
In filename containing, "cleaned" related to the ones that are after the cleaning pre-processing in adjlist generation phases.
I used lemmatization and better tokenization in the ReadInfo.py initialization function.
***
