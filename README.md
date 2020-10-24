# README
## Introduction
* This is the Information Retrieval HW1
* Using TF-IDF to compute the relation between given querys and documents

## Approach
* Term frequence use Log Normalization 
    * <img src="https://latex.codecogs.com/gif.latex?TF=1+log_2(tf_{i,j})"/> 
* Inverse document frequency use Inverse Frequency Smooth
    * <img src="https://latex.codecogs.com/gif.latex?IDF=ln(1+\frac{N}{n_i})"/><br>
        * <img src="https://latex.codecogs.com/gif.latex?N"/>: 1 + total number of documents
        * <img src="https://latex.codecogs.com/gif.latex?n_i"/>: 1 + number of documents that contain the word <img src="https://latex.codecogs.com/gif.latex?w_i "/>
        * plus one to avoid divide zero
* TF-IDF for document and query
    * <img src="https://latex.codecogs.com/gif.latex?TFIDF=TF*IDF"/>