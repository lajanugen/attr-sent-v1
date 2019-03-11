### Preparing training data

The training code expects data to be in TFRecord format. `data_prep/convert.py` can be used to convert text data into this form. An example invocation on the sentiment data from Shen et al. is as below:
```
python data_prep/convert.py data/sentiment.test.0,data/sentiment.test.1 <vocab file> <output dir>
```

### Training and evaluation
The `scripts` folder has bash scripts to train and sample from the models.
