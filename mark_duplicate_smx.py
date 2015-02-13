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
import datetime, re, sys
import numpy as np
import logging
import nltk
import settings


class Command(BaseCommand):

    def handle(self, *args, **options):
        from entry.models import Entry
        similarityThreshold = 0.465
        
        start_date = datetime.now()
        end_date_comparision = start_date - timedelta(hours=48)
        t0= time()
        
        items = Entry.objects.order_by("approved_on").values_list('title', 'body_html', 'id', 'created_on')
        itemList = list(items[:200])
        size = len(itemList)
        storyId = [(cols[2]) for cols in itemList]
        storyBody = [cols[1] for cols in itemList]
        storyTitle = [(cols[0]) for cols in itemList]
        storyCreated = [(cols[3]) for cols in itemList]
        
        english_stemmer = nltk.stem.SnowballStemmer('english')
        class StemmedTfidfVectorizer(TfidfVectorizer):
            def build_analyzer(self):
                analyzer = super(TfidfVectorizer, self).build_analyzer()
                return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        #nltk.pos_tag(tokens)
        
        from sklearn.feature_extraction.text import TfidfVectorizer

        def tokenize_and_stem(text):
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            for token in tokens:
                if(re.search('^[a-zA-Z]+$', token) or re.search('^[0-9.+-]+$', token)):
                    filtered_tokens.append(token)
                    entities = []
                    chunks = nltk.ne_chunk(nltk.pos_tag(token))
                    entities.extend([chunk for chunk in chunks if hasattr(chunk, 'node')])
            stems = [SnowballStemmer("english").stem(t) for t in named_entities]
            lemms = [nltk.WordNetLemmatizer().lemmatize(t) for t in stems]
            return lemms

        print(hasher.get_feature_names()[:-20])
    
        hasher = StemmedTfidfVectorizer(stop_words='english', tokenizer=tokenize_and_stem ,  ngram_range = (1,3), max_df=.99)
        X_body = hasher.fit_transform(storyBody)
        
        thasher = StemmedTfidfVectorizer(stop_words="english", tokenizer=tokenize_and_stem , ngram_range = (1,3), max_df=0.01, binary=True)
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
        nc = int(size*.49)
        km = MiniBatchKMeans(n_clusters= nc, max_iter=100, n_init=5, compute_labels=True, init_size=int(nc*3), batch_size=int(nc*.02), reassignment_ratio=.9, verbose=False).fit(X)
        clusterList = km.labels_
        """
        
        clusterList = [None] * size
        scoreList = [0] * size


        print(size)
        for i in range(size):
            count = 1
            for j in xrange(i,size):
                scoreList[i] += document_distances_lsa_body[i,j] + ((datetime.now() - storyCreated[j]).total_seconds())/(3600*48)
                if((i!=j) and 
                (document_distances_title[i,j]==0 and document_distances_body[i,j]>.70) and 
                (document_distances_title[i,j]>0.4 or document_distances_body[i,j]>.42 or document_distances_lsa_body[i,j]>0.999)):
                            clusterList[j] = i
                            count = count + 1 
                if(clusterList[j] == None):
                    clusterList[j] = j 
        print('\x1b[1;31m'+"Cluster List calculated in %fs" % (time() - t0)+'\x1b[0m')
                
        #creating a dict iterator form the cluster
        clusteredDict = {}
        print(len(clusterList))
        for l in range(len(clusterList)):
            try:
                clusteredDict[clusterList[l]].append([storyId[l],scoreList[l], storyCreated[l]])
            except:
                clusteredDict.setdefault(clusterList[l], [[storyId[l], scoreList[l], storyCreated[l]]])      
        print('\x1b[1;31m'+"Cluster Dict in %fs" % (time() - t0)+'\x1b[0m')


        #Finding root and autotagging
        for cluster  in clusteredDict.values():
            if(len(cluster)>1):
                largest = 0
                for story in cluster:
                    score = story[1]
                    if(score>largest):
                            largest=score
                            root=story[0]
                rootStory = Entry.objects.filter(id=root)
                for story in cluster:
                        duplicate = Entry.objects.filter(id=root)
                        duplicate[0].duplicate_of=rootStory[0]
                        duplicate[0].status = rootStory[0].status
                        duplicate[0].priority = rootStory[0].priority
                        if (duplicate[0].approved_by_id is None) and duplicate[0].status in [2, -1, 5, 6]:
                            duplicate[0].approved_on = datetime.now()
                            duplicate[0].approved_by_id = settings.SYSTEM_ADMIN_USER_ID
                        duplicate[0].primary_topic_id = rootStory[0].primary_topic_id
                        duplicate[0].primary_industry_id = rootStory[0].primary_industry_id
                        duplicate[0].copy_tags_and_sentiment_from(rootStory[0], copyOnlySemanticTags=True)
                        duplicate[0].save(doNotSaveDuplicateChildren=True)
        print('\x1b[1;31m'+"Overall time taken %fs" % (time() - t0)+'\x1b[0m')
