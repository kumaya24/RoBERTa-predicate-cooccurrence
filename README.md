# RoBERTa-predicate-cooccurrence -- Jacqui Wang

## Derivative words based on input
```python word1mask1.py <input.txt> <option> --ft --model <roberta/t5>```

eg. ```python word1mask1.py word_list\vintrans_100.txt nom_vintran -o nominalization\nom_vintran.tsv```

## To finetune the mode
### Roberta:
1. ```python pre-finetuning.py <type-option>```
    -> output: ```ft_datatset\<type-option>_dataset.pt```
2. ```python run_ft.py <type-option> --epochs <num> --batch_size <num> --model roberta ```     

### T5
1. ```python run_ft.py <type-option> --epochs <num> --batch_size <num> --model t5 ```

