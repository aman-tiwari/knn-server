from __future__ import print_function

import tornado
import time

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ujson
import torchfile
from sklearn.neighbors import NearestNeighbors

PORT = 4545
WORKERS = 12 * 2 + 1

# A dict mapping from query params to sample-file filenames
samples_filenames = {frozenset(('Detroit',)) : \
                     '/home/studio/Documents/level19/Detroit_z19_features.t7'}

all_samples = {key : torchfile.load(samples_filenames[key]) \
                for key in samples_filenames}

# A dict mapping from query params to sample label-files
sample_labels = {frozenset(('Detroit',)) : \
                 '/home/studio/Documents/level19/Detroit_z19.txt' }

label_to_idx = {}
all_labels = {}

# Load labels into labels dict and also reverse label to idx mapping
for key, label_file in sample_labels.viewitems():
    with open(label_file, 'r') as f:
        
        labels = [line.rstrip() for line in f.readlines()] #filenames
        print('First 3 labels of ', key, labels[:3], '...', '\n')
        all_labels[key] = labels
        label_to_idx[key] = {label : i for i, label in enumerate(labels)}

# A dict from query params to knn tree
knn_trees = {}

# Insert samples into ball trees
for key, samples in all_samples.viewitems():
    
    ball_tree = NearestNeighbors(algorithm='ball_tree', metric='euclidean')

    print('inserting into', str(key),'tree...')
    m = time.time()
    
    print('First 3 features of first 3 samples:', samples[:3][:3])
    
    neighbours = ball_tree.fit(samples)
    knn_trees[key] = neighbours
    
    print('inserted into tree, took:', str(time.time() - m),'s')

class Handler(RequestHandler):
    
    executor = ThreadPoolExecutor(max_workers=WORKERS)
    
    @run_on_executor
    def do_knn(self, key, search_label, limit):
        try:
            search_idx = label_to_idx[key][search_label]
            search_sample = all_samples[key][search_idx]
        except KeyError:
            return ujson.dumps({'error' : 'invalid request'})
        
        distances, indices = neighbours.kneighbors(search_sample.reshape(1, -1), limit)
        
        labels = all_labels[key]
        matches = [{'distance':dist, 'label':labels[i]} \
                       for dist, i in zip(distances[0].tolist(), indices[0].tolist())] 
        
        res = {'matches' : matches}
        return ujson.dumps(res)

    @tornado.gen.coroutine
    def get(self):
        param = frozenset(tuple(self.get_arguments('param')))
        search_label = self.get_argument('label', '')
        limit = int(self.get_argument('limit', 25))
        res = yield self.do_knn(param, search_label, limit)
        self.write(res)
        
print("Starting server on", str(PORT))
app = Application([("/", Handler)], debug=False)
HTTPServer(app).listen(PORT)
IOLoop.instance().start()
