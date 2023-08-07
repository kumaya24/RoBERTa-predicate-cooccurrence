# RoBERTa-predicate-cooccurrence
## Calculate probability ratios
First, calculate probability ratios using `cooc_roberta.py` by specifying two word lists (one word per line) and their respective types (noun, adj, vtransSubj, vtransObj, or vintrans). For example:

```python cooc_roberta.py adjectives.txt adj nouns.txt noun > ratios.tsv```

## Use ratios to calculate cooccurrence scores
Second, feed the output of the previous step into `table2scores.py` to compute the cooccurrence scores. Continuing from the previous example:

```python table2scores.py < ratios.tsv > scores.tsv```

