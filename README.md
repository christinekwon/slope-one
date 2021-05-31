# Slope One Predictor

## Description

An implementation of a slope one filtering algorithm, an item-baed collaborative filtering method from the 2005 [paper](https://arxiv.org/pdf/cs/0702144.pdf) by Daniel Lemire and Anna Maclachlan. 

** tests to-be-added

## Input

All input is read from stdin, and comes in the following format. All items on a single line are separated by white space.

```bash
user_count item_count
rating_count
user_id item_id rating [from line 2 to rating_count + 1]
...
rec_user
rec_size
```

## Output

Output is written to stdout in the following format. For rec_size lines, each lines contains an item ID and its predicted rating, separated by white space. The results are written in decreasing rating order, and for identical predicted ratings, they are printed in increasing ID order. All ratings are rounded up to the third decimal place

```bash
item_id predicted_rating
...
```

## Usage

```python
python3 demo.py input_*.txt