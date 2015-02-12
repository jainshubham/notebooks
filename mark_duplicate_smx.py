import os, sys
from django.core.management.base import BaseCommand
from optparse import make_option
from datetime import datetime, timedelta
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import logging

import settings


class Command(BaseCommand):

    def handle(self, *args, **options):
        from entry.models import Entry
        similarityThreshold = 0.465
        
        start_date = datetime.now()
        end_date_comparision = start_date - timedelta(hours=48)
        t0= time()
        
        item = Entry.objects.order_by("approved_on")
        itemList = item[:200]
        size = len(itemList)
        print(size)
        storyCols = itemList.values_list('title', 'body_html')
        storyBody = [''.join(cols[1]) for cols in storyCols]
        storyTitle = [''.join(cols[0]) for cols in storyCols]

        hasher = TfidfVectorizer(stop_words='english', token_pattern=u'(?u)[a-zA-Z0-9]+',  ngram_range = (1,3), max_df=.99)
        X_body = hasher.fit_transform(storyBody)
        
        thasher = TfidfVectorizer(stop_words="english", token_pattern=u'(?u)[a-zA-Z0-9]+', ngram_range = (1,3), max_df=0.005)
        X_title = thasher.fit_transform(storyTitle)

        lsa = TruncatedSVD(algorithm='arpack', n_components=150)
        dtm_lsa_body = lsa.fit_transform(X_body)
        X_lsa_body = Normalizer(copy=False).fit_transform(dtm_lsa_body)

        X = csr_matrix(X_title)
        document_distances_title = (X * X.T)
        
        X = csr_matrix(X_body)
        document_distances_body = (X * X.T)
        
        X = csr_matrix(X_lsa_body)
        document_distances_lsa_body = (X * X.T)
        
        print('\x1b[1;31m'+"ML part completed in %fs" % (time() - t0)+'\x1b[0m')
        """
        document_distances = np.maximum(document_distances_title.toarray(), document_distances_body.toarray(), document_distances_lsa_body.toarray())
        nc = int(size*.49)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.9, verbose=False).fit(X)
        clusterList = km.labels_
        """
        
        document_distances = np.maximum(document_distances_title.toarray(), document_distances_body.toarray(), document_distances_lsa_body.toarray())

        clusterList = [None] * size
        scoreList = [0] * size
        print(size)
        for i in range(size):
            count = 1
            for j in range(i,size):
                s = document_distances[i,j]
                scoreList[i] += document_distances[i,j] 
                if(s>similarityThreshold and i!=j):
                        clusterList[j] = i
                        count = count + 1 
                if(clusterList[j] == None):
                    clusterList[j] = j 
        print('\x1b[1;31m'+"Cluster List calculated in %fs" % (time() - t0)+'\x1b[0m')
                
        

        print('\x1b[1;31m'+"Cluster List calculated in %fs" % (time() - t0)+'\x1b[0m')

        #creating a dict iterator form the cluster
        clusteredDict = {}
        scoreDict = {}
        print(len(clusterList))
        for l in range(len(clusterList)):
            try:
                clusteredDict[clusterList[l]].append([itemList[l],scoreList[l]])
            except:
                clusteredDict.setdefault(clusterList[l], [[itemList[l], scoreList[l]]])        
        print('\x1b[1;31m'+"Cluster Dict in %fs" % (time() - t0)+'\x1b[0m')


        #Finding root and autotagging
        for cluster  in clusteredDict.values():
            if(len(cluster)>1):
                largest = 0
                for story in cluster:
                    score = (datetime.now() - story[0].created_on).total_seconds()   
                    score = story[1]
                    #print(story[0].title, score)
                    if(score>largest):
                            largest=score
                            root=story[0]
                for story in cluster:
                    if (story[0].id!=root.id):
                        story[0].duplicate_of=root
                        story[0].status = root.status
                        story[0].priority = root.priority
                        if (story[0].approved_by_id is None) and story[0].status in [2, -1, 5, 6]:
                            story[0].approved_on = datetime.now()
                            story[0].approved_by_id = settings.SYSTEM_ADMIN_USER_ID
                        story[0].primary_topic_id = root.primary_topic_id
                        story[0].primary_industry_id = root.primary_industry_id
                        story[0].copy_tags_and_sentiment_from(root, copyOnlySemanticTags=True)
                        story[0].save(doNotSaveDuplicateChildren=True)
        print('\x1b[1;31m'+"Overall time taken %fs" % (time() - t0)+'\x1b[0m')
