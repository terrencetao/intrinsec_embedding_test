# intrinsec_embedding_test

This repo contains a code for word embedding evaluation. 
you have 3 folders :
         intresect_test : contains the code for intresect properties evaluations of word embedding ( similarity, analogy ...)
         extrinsect_test : contains the code for extresect properties evaluations of word embedding (embedding appply on some task through notebook or .py files )
         models: which contains a word vector derive from OANC corpus for test.

To exploit the intresect_test module you can use this line :
python --model_folder  'folder where is a .pk which contains vectors (dic structure)'       --dim 'dimension of a vectors 50 ,150 ...'  --w2v 'a name of .pk file'

for the extrinsect_test actually you need to check the folder and run each code separately


also you have another repo https://github.com/kudkudak/word-embeddings-benchmarks which achieves the same objective but has constraints.
