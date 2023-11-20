# composite-lang-models
Comparison of classification power of different transformer-based composite language models, as well as their combinations.

this repository is created to ease the replication of results (and further reasearch on the subject) of
[Transformer-Based Composite Language Models for Text Evaluation and Classification](https://doi.org/10.3390/math11224660)

If you reuse the code or the data prepared and produced, please cite: [https://doi.org/10.3390/math11224660](https://doi.org/10.3390/math11224660)

Abstract: Parallel natural language processing systems were previously successfully tested on the tasks of part-of-speech tagging and authorship attribution through mini-language modeling, for which they achieved significantly better results than independent methods in the cases of seven European languages. The aim of this paper is to present the advantages of using composite language models in the processing and evaluation of texts written in arbitrary highly inflective and morphology-rich natural language, particularly Serbian. A perplexity-based dataset, the main asset for the methodology assessment, was created using a series of generative pre-trained transformers trained on different representations of the Serbian language corpus and a set of sentences classified into three groups (expert translations, corrupted translations, and machine translations). The paper describes a comparative analysis of calculated perplexities in order to measure the classification capability of different models on two binary classification tasks. In the course of the experiment, we tested three standalone language models (baseline) and two composite language models (which are based on perplexities outputted by all three standalone models). The presented results single out a complex stacked classifier using a multitude of features extracted from perplexity vectors as the optimal architecture of composite language models for both tasks.

in order to reproduce the results:

(0. install python and the required packages e.g. transformers and torch)

All of the data required to recreate the results as well as the derived files and results themselves are located in [data](https://github.com/procesaur/composite-lang-models/tree/main/data) directory.
if you do not want to recreate the derivative files and results skip steps 1 and 2.

1. run [data_preprocessing.py](https://github.com/procesaur/composite-lang-models/blob/main/data_preprocessing.py) to recreate the derivative files from source perplaxity-based dataset;
2. run [test.py](https://github.com/procesaur/composite-lang-models/blob/main/test.py) to recreate the result files for both standalone and composite models from derivative files produced in step 1;
3. run [results.py](https://github.com/procesaur/composite-lang-models/blob/main/results.py) to print readable accuracy results for each model, using the resutls file produced in step 2.

5. (optional) run [pearson.py](https://github.com/procesaur/composite-lang-models/blob/main/pearson.py) to print the pearson corelations between both models and datasets.
