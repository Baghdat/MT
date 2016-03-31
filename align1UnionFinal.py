#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import os
import subprocess
import math
#import matplotlib.pyplot as plt
import operator
optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1 MODEL...")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
for (n, (f, e)) in enumerate(bitext):
	for f_i in set(f):
		f_count[f_i] += 1
		for e_j in set(e):
			fe_count[(f_i,e_j)] += 1
	for e_j in set(e):
		e_count[e_j] += 1

t = dict()

for (f_i, e_j) in fe_count.keys():
	t[(f_i, e_j)] = 1/float(len(e_count.keys()))   #initialize t(f|e) uniformly
#for (f_i, e_j) in fe_count.keys():
#  theta_prior[(f_i,e_j)] = fe_count[(f_i, e_j)]/float(e_count[e_j])

num_iteration = 20
likelihood  = []
for iteration in range(num_iteration):
	
	count = defaultdict(float) # partial count of English word aligned to German
	total = defaultdict(float) #total count of English word aligned to German

  #Expectation step 
	for (n, (f, e)) in enumerate(bitext):   #for all sentence pairs (e, f) do
		for f_i in set(f):                     #for all words f_i in f do
			s_total = 0.0                        #s-total(f)=0
			for e_j in set(e):                   #for all words e_j in e do
				s_total += t[(f_i, e_j)]  
			for e_j in set(e):                        #for each <f,e> in count do
				#for f_i in set(f):
				count[(f_i, e_j)] += t[(f_i, e_j)]/float(s_total)                   #count[<f(i)^(n), e(j)^(n)>]++ 
				total[e_j] += t[(f_i, e_j)]/float(s_total)                          #count[e(j)^n]++
  #Maximization step
	for (f, e) in count.keys():
		t[(f,e)] = (count[(f, e)])/float(total[e])  ##p(f|e) = count(<f, e>)/count(e)


t2 = dict()
for ( e_j, f_i) in fe_count.keys():
	t2[(f_i, e_j)] = 1/float(len(f_count.keys()))


for iteration in range(num_iteration):
	
	count = defaultdict(float) # partial count of English word aligned to German
	total = defaultdict(float) #total count of English word aligned to German

  #Expectation step 
	for (n, (e,f)) in enumerate(bitext):   #for all sentence pairs (e, f) do
		for f_i in set(f):                     #for all words f_i in f do
			s_total = 0.0                        #s-total(f)=0
			for e_j in set(e):                   #for all words e_j in e do
				s_total += t2[(f_i, e_j)]  
			for e_j in set(e):                        #for each <f,e> in count do
				#for f_i in set(f):
				count[(f_i, e_j)] += t2[(f_i, e_j)]/float(s_total)                   #count[<f(i)^(n), e(j)^(n)>]++ 
				total[e_j] += t2[(f_i, e_j)]/float(s_total)                          #count[e(j)^n]++
  #Maximization step
	for (f, e) in count.keys():
		t2[(f,e)] = (count[(f, e)])/float(total[e])  ##p(f|e) = count(<f, e>)/count(e)






for (n, (f, e)) in enumerate(bitext):
	for (i, f_i) in enumerate(f):
		max_p = 0
		max_p2 = 0
		best_align = 0
		best_align2 = 0
		for (j, e_j) in enumerate(e):
			if (t[(f_i,e_j)] > max_p) or  (t2[(e_j, f_i)] > max_p2):
				max_p2 = t2[(e_j,f_i)]
				max_p = t[(f_i,e_j)]
				best_align = j
		sys.stdout.write("%i-%i " % (i,best_align))
	sys.stdout.write("\n")
