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

import settings


class Command(BaseCommand):


    def handle(self, *args, **options):
        from entry.models import Entry

        t0= time()
        item = Entry.objects.order_by("id")[:1000]  # Note: Can't filter by approved on or approved_by as this is being modified in the code
        Entry.objects.update(duplicate_of=None)
        #storySoup = item.values_list('body_html', flat=True) 
        storyCols = item.values_list('title', 'title', 'title', 'body_html')        
        storySoup = [' '.join(cols) for cols in storyCols] 

        #Vectorizer
        hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
        X = vectorizer.fit_transform(storySoup)
        print("n_samples: %d, n_features: %d" % X.shape)
        print('\x1b[1;31m'+"Stories Vectorized in %fs" % (time() - t0)+'\x1b[0m')

        #Dimension Reduction
        lsa = TruncatedSVD(algorithm='arpack', n_components=250)
        dtm_lsa = lsa.fit_transform(X)
        X = Normalizer(copy=False).fit_transform(dtm_lsa)
        print("n_samples: %d, n_features: %d" % X.shape)
        print('\x1b[1;31m'+"Dimensions Reduced in %fs" % (time() - t0)+'\x1b[0m')


        # Do the actual clustering
        nc = int(len(storySoup)*.95)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.1, verbose=False).fit(X)
        d = km.transform(X)
        #km = KMeans(n_clusters=nc, init='k-means++', max_iter=100, n_init=10, verbose=True).fit(X)
        print('\x1b[1;31m'+"Stories Clustered in  in %fs" % (time() - t0)+'\x1b[0m')
        
        
        # Finding root and autotagging
        for k in np.unique(km.labels_):
            if(not math.isnan(k)):
                members = np.where(km.labels_ == k)
            if k == -1:
                #print("outliers:")
                continue    
            else:
                pass
            largest = 0
            root=None
            print("\n")
            for it in members[0]:
                score = 0
                currItem = item[it]
                score += (((datetime.now() - currItem.created_on).total_seconds() // 3600)-190)/24 # Weight for story age
                score += min(d[it])     # score from clustering
                currItem.score = score
                if(len(members[0])>1 and len(members[0])<5):
                    print(currItem.title)
                if(score>largest):
                        largest=score
                        root=currItem
            for it in members[0]:
                currItem = item[it]
                if (currItem.id!=root.id):
                    currItem.duplicate_of=root
                    currItem.status = root.status
                    currItem.priority = root.priority
                    if (currItem.approved_by_id is None or currItem.approved_by_id == settings.SYSTEM_ADMIN_USER_ID) and currItem.status in [2, -1, 5, 6]:
                        currItem.approved_on = datetime.now()
                        currItem.approved_by_id = settings.SYSTEM_ADMIN_USER_ID
                    currItem.primaryl_topic_id = root.primary_topic_id
                    currItem.primary_industry_id = root.primary_industry_id
                    currItem.copy_tags_and_sentiment_from(root, copyOnlySemanticTags=True)
                    currItem.save(doNotSaveDuplicateChildren=True)
        print('\x1b[1;31m'+"Overall time taken %fs" % (time() - t0)+'\x1b[0m')
