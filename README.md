# Swiss-German LID using BERT

This repository let's you finetune a BERT model to perform the task of Language Identification.
The target task is to properly identify Swiss German. 

The languages the model will be trained on are:

* `afr`: Afrikaans
* `deu`: German
* `gsw`: Swiss German
* `gsw_like`: a mix of Bavarian, Kolsch, Limburgan, Low German, Northern Frisian and Palatine German
* `ltz`: Luxembourgian
* `nld`: Dutch
* `other`: a mix of Catalan, Croatian, Danish, Esperanto, Estonian, Finnish, French, Irish, Galician,
Icelandic, Italian, Javanese, Konkani, Papiamento, Portuguese, Romanian, Slovenian, Spanish, Swahili and Swedish

The procedure:

1. install this repo:  `python setup.py install` (or `python setup.py develop`, which is equivalent to `pip install -e`);
2. get Swiss German sentences into a CSV file;
3. use the scripts in `training` to generate a model (see below);
4. set the generated model as a default in the module `bert_lid`, by copying the out directory to `bert_lid/models/default`;
   (Note: if you didn't install the module in development mode, the model must be written to the location of the installed module);
5. now, you can use `bert_lid.BertLid` and install it in other environments;

## Training a model

<p style="background-color: #FF000055">
**Important notice** we provide everything needed to train the model, **except the Swiss-German** data.
It is your task to generate one CSV file containing Swiss German sentences in a column named `text`.
Tip: you can access Swiss German sentences from the Leipzig Corpora Collection.
</p>

Once you have a Swiss German CSV file ready, the only thing left to do is to run the scripts in the `training` folder in order.

```bash
# ensure you launch the scripts from the training directory !
cd training

./1_download-data.py 
./2_prepare-data.py --gsw path/to/swiss-german-sentences.csv
./3_split-data.py
./4_finetune-bert.sh  # <= this one long-running (>20 minutes), would better be running in a screen
./4_eval-model.py
```

At this point, you should have a model saved in `training/out`. The only thing left to do is to make it the default model,
by copying it to `bert_lid/models/default` (actual location varies depending on the kind of installation you did, install or development):

```bash
bert_lid_install_model -i training/out
```

## Inference

As long as you have a model somewhere (or installed one as the default using script 6 in `training`), this is straight-forward:

```python
>>> from bert_lid import BertLid
>>> lid = BertLid()
>>> lid.predict(['Das isch sone seich'])
(['gsw'], [99.83619689941406])
>>> lid.predict(['Trop top ce module, il marche bien et est bien documenté!'], mode='row')
[('Trop top ce module, il marche bien et est bien documenté!', 'other', 99.75711822509766)]
```

