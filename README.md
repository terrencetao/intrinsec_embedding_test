# Word Embedding Evaluation

This repository contains code for evaluating word embeddings using both intrinsic and extrinsic properties. The evaluation is organized into three main folders:

## Folder Structure

   1. intrinsect_test: Intrinsic evaluation code (similarity, analogy, etc.)
   2. extrinsect_test: Extrinsic evaluation code (tasks applied through notebooks or .py files)
   3. models: Pre-trained word vectors from the OANC corpus
# Usage
## Intrinsic Evaluation

To exploit the `intrinsect_test` module, you can use the following command:
1. Navigate to the intrinsect_test folder.
2. Run the evaluation script with the appropriate arguments:
```sh
python --model_folder 'folder where the .pk file containing vectors (dictionary structure) is located' --dim 'dimension of the vectors (e.g., 50, 150, etc.)' --w2v 'name of the .pk file'
python --model_folder 'path_to_model_folder' --dim 'vector_dimension' --w2v 'vector_file_name.pk'```

## Extrinsic Evaluation

1. Navigate to the extrinsect_test folder.
2. Check and run each code file separately as needed.



Ensure you have the necessary dependencies installed. Refer to each module's specific requirements and install them accordingly.
## Alternative Repository

Additionally, you may want to check out another repository that achieves the same objective but with certain constraints: word-embeddings-benchmarks.
Folder Structure

   


    
