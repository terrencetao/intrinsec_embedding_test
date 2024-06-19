# Word Embedding Evaluation

This repository contains code for evaluating word embeddings using both intrinsic and extrinsic properties. The evaluation is organized into three main folders:

1. `intrinsect_test`: Contains the code for evaluating intrinsic properties of word embeddings such as similarity and analogy.
2. `extrinsect_test`: Contains the code for evaluating extrinsic properties of word embeddings by applying them to specific tasks through notebooks or `.py` files.
3. `models`: Contains word vectors derived from the OANC corpus for testing purposes.

## Intrinsic Evaluation

To exploit the `intrinsect_test` module, you can use the following command:

```sh
python --model_folder 'folder where the .pk file containing vectors (dictionary structure) is located' --dim 'dimension of the vectors (e.g., 50, 150, etc.)' --w2v 'name of the .pk file'
