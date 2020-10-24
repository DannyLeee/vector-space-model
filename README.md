# README
## Introduction
* This is the Information Retrieval HW1
* Using TF-IDF to compute the relation between given querys and documents

## Approach
* Term frequence use Log Normalization 
    * $TF = 1 + log_2(tf_{i,j})$
* Inverse document frequency use Inverse Frequency Smooth
    * $IDF = ln(1 + \frac{N}{n_i})$
        * $N$: 1 + total number of documents
        * $n_i$: 1 + number of documents that contain the word $w_i$
        * plus one to avoid divide zero
* TF-IDF for document and query
    * $TFIDF = TF * IDF$