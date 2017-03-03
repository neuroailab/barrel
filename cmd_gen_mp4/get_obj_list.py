'''
From 3D world object database, get the list of objects and then try to get a balanced object list
'''

import pymongo
import time
from bson.objectid import ObjectId
import numpy as np

conn = pymongo.MongoClient(port=22334)
coll = conn['synthetic_generative']['3d_models']

#test_coll_  = coll.find({'type': 'shapenetremat', 'has_texture': {'$exists': False}})
test_coll  = coll.find({'type': 'shapenetremat', 'has_texture': True})
#print(test_coll.count())
#print(test_coll[0])

cached_coll     = list(test_coll)

synset_dict     = {}

for indx, item in enumerate(cached_coll):
    if item['shapenet_synset'] not in synset_dict:
        synset_dict[item['shapenet_synset']] = []
    synset_dict[item['shapenet_synset']].append(indx)

obj_num     = len(synset_dict.keys())
print(obj_num)
len_of_obj  = [len(synset_dict[key_tmp]) for key_tmp in synset_dict]
len_of_obj.sort()
print(len_of_obj)

all_sum     = 10000
final_list  = []
for indx_tmp, item_tmp in enumerate(len_of_obj):
    if sum(final_list) + (obj_num - indx_tmp)*item_tmp > all_sum:
        fill_num    = (all_sum - sum(final_list)) // (obj_num - indx_tmp)
        for _not_used in xrange(indx_tmp, obj_num):
            final_list.append(fill_num)
        break
    else:
        final_list.append(item_tmp)

print(len(final_list))
print(final_list)
print(sum(final_list))
print(np.std(final_list))
