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
        discardedStories = 0
        #item = Brief.objects.values()
        item = Entry.objects.order_by("id")[:4800]
        #tempSoup= str(item[i].title+item[i].section+item[i].topic+item[i].industry+item[i].body_html+item[i].primary_topic+item[i].primary_industry+item[i].auto_tagged_topic+item[i].auto_tagged_industry)
        storySoup = item.values_list('body_html', flat=True)
        storyTitle = item.values_list('title', flat=True)
        storyId = item.values_list('id', flat=True)
        storyPriority = item.values_list('priority', flat=True)
        storyDuplicate = item.values('duplicate_of')
        storyScore= item.values_list('score')
        
        
        #Vectorizer
        hasher = HashingVectorizer(stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
        X = vectorizer.fit_transform(storySoup)
        print("n_samples: %d, n_features: %d" % X.shape)
        print("n_samples: %d, n_features: %d" % X.shape)
        print('\x1b[1;31m'+"Stories Vectorized in %fs" % (time() - t0)+'\x1b[0m')

        #Dimension Reduction
        lsa = TruncatedSVD(algorithm='arpack', n_components=250)
        dtm_lsa = lsa.fit_transform(X)
        X = Normalizer(copy=False).fit_transform(dtm_lsa)
        print(len(storySoup))
        print('\x1b[1;31m'+"Dimensions Reduced in  in %fs" % (time() - t0)+'\x1b[0m')
        print("n_samples: %d, n_features: %d" % X.shape)

        # Do the actual clustering
        nc = int(len(storySoup)*.75)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.1, verbose=False).fit(X)
        d = km.transform(X)
        #km = KMeans(n_clusters=nc, init='k-means++', max_iter=100, n_init=10, verbose=True).fit(X)
        labels = km.labels_
        labels_true = storyPriority
        labels_pred = km.labels_
        print('\x1b[1;31m'+"Stories Clustered in  in %fs" % (time() - t0)+'\x1b[0m')
        
        
        clustered={}
        for k in np.unique(km.labels_):
            if(not math.isnan(k)):
                members = np.where(km.labels_ == k)  #[1]
            if k == -1:
                #print("outliers:")
                continue    
            else:
                pass
            largest = 0
            print("\n")
            for it in members[0]:
                score = 0
                currItem = item[it]
                #score += (6-Entry.intra_publication_status_precedence.index(entry.status))/float(6)   #intra_publication_status_precedence
                score += (((datetime.now() - currItem.created_on).total_seconds() // 3600)-190)/24 # Weight for story age
                score += min(d[it])  
                currItem.score = score
                #if(len(members[0])>1 and len(members[0])<5):
                #    print(currItem.title)
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
                    currItem.primary_topic_id = root.primary_topic_id
                    currItem.primary_industry_id = root.primary_industry_id
                    currItem.copy_tags_and_sentiment_from(root, copyOnlySemanticTags=True)
                    currItem.save()
        print('\x1b[1;31m'+"Overall time taken %fs" % (time() - t0)+'\x1b[0m')
