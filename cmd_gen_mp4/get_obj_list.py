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
print(test_coll[0])

cached_coll     = list(test_coll)

synset_dict     = {}
wanted_name     = 'synset'

for indx, item in enumerate(cached_coll):
    if item[wanted_name][0] not in synset_dict:
        synset_dict[item[wanted_name][0]] = []
    synset_dict[item[wanted_name][0]].append(indx)

wanted_name     = 'shapenet_synset'
shapenet_synset_dict = {}

for indx, item in enumerate(cached_coll):
    if item[wanted_name] not in shapenet_synset_dict:
        shapenet_synset_dict[item[wanted_name]] = {}
        shapenet_synset_dict[item[wanted_name]]['listindx'] = []
        shapenet_synset_dict[item[wanted_name]]['listname'] = []

    shapenet_synset_dict[item[wanted_name]]['listindx'].append(indx)
    if item['synset'][0] not in shapenet_synset_dict[item[wanted_name]]['listname']:
        shapenet_synset_dict[item[wanted_name]]['listname'].append(item['synset'][0])

#for key_now in synset_dict:
#    print key_now, len(synset_dict[key_now])

for big_key in shapenet_synset_dict:
    print big_key,
    print len(shapenet_synset_dict[big_key]['listindx']),
    for small_key in shapenet_synset_dict[big_key]['listname']:
        print "%s %i" % (small_key, len(synset_dict[small_key])),
    print

print(cached_coll[synset_dict['n20000038'][0]])

'''
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
'''
