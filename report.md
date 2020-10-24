<p style="text-align:right;">
姓名:李韋宗<br>
學號:B10615024<br>
日期:2020/10/29<br>
</p>

<h1 style="text-align:center;"> Homework 1: Vector Space Model

## 建置環境與套件
* Python 3.6.9, sklearn.metrics.pairwise.cosine_similarity, pandas, numpy, collections. Counter, tqdm.tqdm, datetime.datetime, timezone, timedelta

## 資料前處理
* 按照 doc_list.txt/query_list.txt 的順序讀入文章及 query
* 將所有文章存成 list of string 方便加速後續 TF 的計算(集中 file I/O 的時間)
* 將所有計算好的 query TF 存成 DataFrame
    * DataFrame 的 columns 就是字典(減少重複讀取)

## 模型參數調整
* Tf 使用 Log Normalization 
    * $TF = 1 + log_2(tf_{i,j})$
* IDF 使用 Inverse Frequency Smooth 由文章的 TF 得出
    * $IDF = ln(1 + \frac{N}{n_i})$
* 文章與 query 的 TF-IDF 都直接相乘
    * $TFIDF = TF * IDF$

## 模型運作原理
* 為加速運算(偷吃步)，將單字表壓縮在 query 所出現的字
* 讓 TF, IDF, TFIDF 的維度都剩下 123 維

## 心得
* 最開始先用 sklearn 的套件計算 TFIDF，但因理解錯誤讓 IDF 包含 query 所出現的次數，導致無法突破 baseline，後來更改參數後得到較高的成績，便動手實作重現套件功能。起初直接操作 DataFrame 計算文章的 TF，但效率低落，後來用 list of dict 再轉存加速不少，重現出套件的操作成果後，總執行時間將近 2 分鐘。跟同學討論後，使用上述的方法再次加速，讓總執行時間剩下將近 3 秒鐘，速率瓶頸為讀完所有文章轉成 list fo sting，也就是 file I/O 的硬體限制；意外的是使用這個方法不只速度快，同樣的參數設定，在資料集上的分數也提高了，我想是因為只拿 query 所出現的單字這樣偷吃步的方法，猶如機器學習先看過 testing data 一樣，分數自然會提高。



<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>