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
        start_date = datetime.now()
        end_date_comparision = start_date - timedelta(hours=250)
        t0= time()
        item = Entry.objects.order_by("id").filter(created_on__gte=end_date_comparision, created_on__lte=start_date)
        item.update(duplicate_of=None)
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
        nc = int(len(storySoup)*0.95)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.1, verbose=False).fit(X)
        d = km.transform(X)
        #km = KMeans(n_clusters=nc, init='k-means++', max_iter=100, n_init=10, verbose=True).fit(X)
        print('\x1b[1;31m'+"Stories Clustered in  in %fs" % (time() - t0)+'\x1b[0m')

        #creating a dict iterator form the cluster
        clusteredDict = {}
        for l in range(len(km.labels_)):
            #item[l].score = score
            try:
                clusteredDict[km.labels_[l]].append(item[l])
            except:
                clusteredDict.setdefault(km.labels_[l], [item[l]])

        #Finding root and autotagging
        for cluster in clusteredDict.values():
            #print("\n")
            largest = 0
            for story in cluster:
                score = (datetime.now() - story.created_on).total_seconds()      # Weight for story age
                #score += min(d[it])     # score from clustering
                #story.score = score
                #if(len(cluster)>1 and len(cluster)<5):
                #   print(story.title)
                if(score>largest):
                        largest=score
                        root=story
            for story in cluster:
                if (story.id!=root.id):
                    story.duplicate_of=root
                    story.status = root.status
                    story.priority = root.priority
                    if (story.approved_by_id is None) and story.status in [2, -1, 5, 6]:
                        story.approved_on = datetime.now()
                        story.approved_by_id = settings.SYSTEM_ADMIN_USER_ID
                    story.primary_topic_id = root.primary_topic_id
                    story.primary_industry_id = root.primary_industry_id
                    story.copy_tags_and_sentiment_from(root, copyOnlySemanticTags=True)
                    story.save(doNotSaveDuplicateChildren=True)
        print('\x1b[1;31m'+"Overall time taken %fs" % (time() - t0)+'\x1b[0m')
