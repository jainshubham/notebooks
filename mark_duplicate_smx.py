import os, sys
from django.core.management.base import BaseCommand
from optparse import make_option
from datetime import datetime, timedelta
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
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
        itemList = item[:2000]
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

        clusterList = [None] * size
        cluster = 0
        for i in range(size):
            for j in range(i,size):
                s  = max(document_distances_title[i, j], document_distances_body[i, j], document_distances_lsa_body[i,j]/2)
                if(s>similarityThreshold and i!=j):
                        #print(storyTitle[i])
                        #print(storyTitle[j])
                        #print(s, i, j)
                        clusterList[j] = cluster
            if(clusterList[i] == None):
                clusterList[i] = cluster    
            cluster += 1  
        print('\x1b[1;31m'+"Cluster List calculated in %fs" % (time() - t0)+'\x1b[0m')
        
 
    
        #creating a dict iterator form the cluster
        clusteredDict = {}
        print(len(clusterList))
        for l in range(len(clusterList)):
            try:
                clusteredDict[clusterList[l]].append(itemList[l])
            except:
                clusteredDict.setdefault(clusterList[l], [itemList[l]])
        print('\x1b[1;31m'+"Cluster Dict in %fs" % (time() - t0)+'\x1b[0m')

        #Finding root and autotagging
        for cluster in clusteredDict.values():
            if(clusterList[l]!=-1):
                largest = 0
                for story in cluster:
                    score = (datetime.now() - story.created_on).total_seconds()     
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
