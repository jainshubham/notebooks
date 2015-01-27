import os, sys
from django.core.management.base import BaseCommand
from optparse import make_option
from datetime import datetime, timedelta
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
import logging
import numpy as np
import pandas as pd
from random import random
import nltk
import math


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--comparision_range', default="48", dest='comparision_range',
            help='By default comparisions will be made to stories saved in last 48 hours'),
        
        make_option('--story_duplicate_range', default="16", dest='story_duplicate_range',
            help='By default stories saved in last 16 minutes will be analyzed'),
        make_option('--by_time_range', default="N", dest='by_time_range',
            help="""
            Analyze stories in given time frame, format eg: 'Day|H:M:S-H:M:S'
            """),
    )
    def __init__(self, *args, **kwargs):
        super(Command, self).__init__(*args, **kwargs)
        self.duplicates = []
        
    def handle(self, *args, **options):
        from entry.models import Entry
        t0= time()
        discardedStories = 0
        item = Entry.objects.order_by("-approved_on")[:1200]
        #storySoup= str(item[i].title+item[i].section+item[i].topic+item[i].industry+item[i].body_html+item[i].primary_topic+item[i].primary_industry+item[i].auto_tagged_topic+item[i].auto_tagged_industry)
        storySoup = item.values_list('body_html', flat=True)
        storyTitle = item.values_list('title', flat=True)
        storyId = item.values_list('id', flat=True)

        #Vectorizer
        hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
        X = vectorizer.fit_transform(storySoup)
        print("n_samples: %d, n_features: %d" % X.shape)
        print('\x1b[1;31m'+"Stories Vectorized in %fs" % (time() - t0)+'\x1b[0m')

        #Dimension Reduction
        lsa = TruncatedSVD(algorithm='arpack', n_components=200)
        dtm_lsa = lsa.fit_transform(X)
        X = Normalizer(copy=False).fit_transform(dtm_lsa)
        print(len(storySoup))
        print('\x1b[1;31m'+"Dimensions Reduced in  in %fs" % (time() - t0)+'\x1b[0m')
        print("n_samples: %d, n_features: %d" % X.shape)

        # Do the actual clustering
        nc = int(len(storySoup)*.50)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.1, verbose=False).fit(X)
        #km = KMeans(n_clusters=nc, init='k-means++', max_iter=100, n_init=10, verbose=True).fit(X)
        labels = km.labels_
        labels_pred = km.labels_
        print('\x1b[1;31m'+"Stories Clustered in  in %fs" % (time() - t0)+'\x1b[0m')

        # score
        for k in np.unique(km.labels_):
            if(not math.isnan(k)):
                members = np.where(km.labels_ == k)  
            if k == -1:
                #print("outliers:")
                continue    
            else:
                pass
            for item in members[0]:
                storyScore[item]=random(0,1)

                    
          #updateCluster


