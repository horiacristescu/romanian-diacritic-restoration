from collections import defaultdict
import numpy as np

def isLOL(x):
    if type(x) in (list, tuple):
        if (len(x)>0 and type(x[0]) in (list, tuple)) or (len(x)==0):
            return True
    return False

def LOL_get(x, col):
    if not isLOL(x):
        raise Exception("Err in LOL0")
    return [ pair[col] for pair in x ]

def LOL_keys(x):
    if not isLOL(x):
        raise Exception("Err in LOL0")
    return [ pair[0] for pair in x ]

def LOL_astext(x, sep=" "):
    return sep.join( LOL_keys(x) )

def LOL_values(x):
    if not isLOL(x):
        raise Exception("Err in LOL0")
    return [ pair[1] for pair in x ]

def LOL_sort(x, col=1, reverse=True):
    if type(x) in [dict, defaultdict]:
        x = x.items()
    if type(x)==list and len(x)==0:
        return x
    if not isLOL(x):
        raise Exception("Err in LOLsort")
    return sorted(x, key=lambda pair: pair[col], reverse=reverse)

def LOL_norm(x):
    if len(x) == 0:
        return x
    xmax = max(LOL_values(x))
    if xmax>0:
        x = [ (w, round(s/xmax,3)) for w, s in x ]
    return x

def LOL_norm_sum(x):
    if len(x)==0:
        return x
    xsum = sum(LOL_get(x,1))
    if xsum > 0:
        x = [ (w, round(s/xsum,3)) for w,s in x ]
    else:
        raise Exception('LOL_norm_sum xsum negativ')
    return x

def LOL_treshold(l, prag=0, subtract=False):
    if subtract:
        return [ (w,s-prag) for w,s in l if s >= prag]
    else:
        return [ (w,s) for w,s in l if s >= prag]

def LOL_sim_rank(cloud, vec, words=None):
    cloud = list(cloud)
    vecs = words[cloud]
    sims = np.dot(vecs, vec)
    rez = LOL_sort(zip(cloud, sims))
    return rez

def LOL_dot_product(l1, l2, min_value=0.05, factor=3.):
    dl1 = defaultdict(float)
    dl2 = defaultdict(float)
    for k,s in l1:
        dl1[k] += s
    for k,s in l2:
        dl2[k] += s
    min_v1 = min_value
    min_v2 = min_value
    if len(dl1)>0:
        min_v1 = min( dl1.values() ) / factor
        min_v1 = min(min_v1, min_value)
    if len(dl2)>0:
        min_v2 = min( dl2.values() ) / factor
        min_v2 = min(min_v2, min_value)
    rez = defaultdict(float)
    for k in set( dl1.keys() + dl2.keys() ):
        v1 = max(dl1.get(k, min_v1), min_v1)
        v2 = max(dl2.get(k, min_v2), min_v2)
        rez[k] = v1 * v2
    return [ (w, round(s, 3)) for w,s in rez.items() ]

def LOL_sum(*lists):
    rez = defaultdict(float)
    for l in lists:
        for k, s in l:
            rez[k] += s
    return LOL_round(LOL_sort(rez.items()))

def LOL_round(l):
    return [ (w, round(s, 3)) for w,s in l ]

def LOL_min_max_value(l, min_val=None, max_val=None):
    if min_val != None:
        l = [ (w, max(s, min_val)) for w,s in l ]
    if max_val != None:
        l = [ (w, min(s, max_val)) for w,s in l ]
    return l

def LOL_str(l):
    return " ".join([ str(w) for w,s in l ])

def isListOfStr(x):
    return type(x)==list and len(x)>0 and type(x[0])==str

def list_pairwise_rank(l1, l2, words=None):
     dp = np.dot(words[l1], words[l2].transpose())
     return LOL_round(LOL_sort(zip(l1, np.sum(dp, axis=1))))

def LOL_mult_scalar(l, alpha=1.):
    return [ (x,s*alpha) for x,s in l ]

def LOL_invert(l):
    smin = None
    smax = None
    for kw,s in l:
        if smin == None or smin > s:
            smin = s
        if smax == None or smax < s:
            smax = s
    print "smin=", smin
    print "smax=", smax
    dif = smax-smin
    if dif==0:
        return l
    return LOL_sort([ (kw, round(  1. - (s-smin)/dif ,3)) for kw,s in l ] )
