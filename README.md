# music-alignment

To get started, first clone a copy of the Bach WTC scores:

```
https://github.com/jthickstun/bach-wtc.git
```

You'll also need a copy of the MAESTRO (v2.0) dataset, [available here](https://magenta.tensorflow.org/datasets/maestro#v200).

After downloading the scores and MAESTRO dataset, you can extract the aligmnent dataset
by calling the `extract` script from the root of this repository:

```
python3 extract.py {path-to-scores}/bach-wtc/ {path-to-maestro}/maestro-v2.0.0
```

You can then explore alignments by viewing the align.ipynb notebook.
