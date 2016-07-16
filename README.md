# knn-server
---
A concurrent kNN server. Uses scikit-learn's Ball Trees to speed up kNN search. 

Loads feature vectors from `.t7` files, although it would be trivial to modify to use other file formats.

Expects features to have corresponding labels in a `.txt`, with one label on each line. The nth `label` corresponds to the nth `feature`.

### Requires
* scikit-learn
* tornado
* torchfile
* ujson
* futures (`pip install futures` if you are using Python 2.x)