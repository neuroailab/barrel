'''
From 3D world object database, get the list of objects and then try to get a balanced object list
'''

import pymongo
import time
from bson.objectid import ObjectId
import numpy as np
from nltk.corpus import wordnet as wn
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='The script to get object category list')
    parser.add_argument('--minobjnum', default = 60, type = int, action = 'store', help = 'Minimun number of one category')

    args    = parser.parse_args()

    conn = pymongo.MongoClient(port=22334)
    coll = conn['synthetic_generative']['3d_models']

    #test_coll_  = coll.find({'type': 'shapenetremat', 'has_texture': {'$exists': False}})
    #test_coll  = coll.find({'type': 'shapenetremat', 'has_texture': True})
    test_coll  = coll.find({'type': 'shapenet', 'version': 2})
    print(test_coll.count())
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
        #if item['synset'][0] not in shapenet_synset_dict[item[wanted_name]]['listname']:
        #    shapenet_synset_dict[item[wanted_name]]['listname'].append(item['synset'][0])

    #for key_now in synset_dict:
    #    print key_now, len(synset_dict[key_now])

    print(len(shapenet_synset_dict))

    '''
    for big_key in shapenet_synset_dict:
        print big_key, wn._synset_from_pos_and_offset('n',int(big_key[1:])),
        print len(shapenet_synset_dict[big_key]['listindx']),
        for small_key in shapenet_synset_dict[big_key]['listname']:
            print "%s %i" % (small_key, len(synset_dict[small_key])),
        print

    print(cached_coll[synset_dict['n20000038'][0]])
    '''

    all_synset_dict = {}
    deepest_synset_dict = {}
    depth_delta_dict = {}

    for indx, item in enumerate(cached_coll):
        tmp_tree_synset = item['synset_tree']

        tmp_deepest_synsets = None
        tmp_deepest_depth = None

        tmp_depth_list  = []
        tmp_exist_synset_list = []

        for tmp_synset in tmp_tree_synset:
            try:
                curr_synset     = wn._synset_from_pos_and_offset('n', int(tmp_synset[1:]))
                curr_depth      = curr_synset.min_depth()

                if tmp_deepest_synsets is None or tmp_deepest_depth < curr_depth:
                    tmp_deepest_synsets = [tmp_synset]
                    tmp_deepest_depth = curr_depth

                elif tmp_deepest_depth==curr_depth:
                    tmp_deepest_synsets.append(tmp_synset)

                if tmp_synset not in all_synset_dict:
                    all_synset_dict[tmp_synset] = {}
                    all_synset_dict[tmp_synset]['list_indx'] = []
                    all_synset_dict[tmp_synset]['exist_son'] = []
                all_synset_dict[tmp_synset]['list_indx'].append(indx)

                tmp_depth_list.append(curr_depth)
                tmp_exist_synset_list.append(tmp_synset)
            except:
                pass

        '''
        if item['shapenet_synset'] not in tmp_exist_synset_list:
            print('Shapenet synset have problem!')
            continue
        curr_depth_delta = tmp_deepest_depth - tmp_depth_list[tmp_exist_synset_list.index(item['shapenet_synset'])]
        if curr_depth_delta not in depth_delta_dict:
            depth_delta_dict[curr_depth_delta] = []
        depth_delta_dict[curr_depth_delta].append(indx)
        '''

        for tmp_deepest_synset in tmp_deepest_synsets:
            if tmp_deepest_synset not in deepest_synset_dict:
                deepest_synset_dict[tmp_deepest_synset] = {}
                deepest_synset_dict[tmp_deepest_synset]['list_indx'] = []
                deepest_synset_dict[tmp_deepest_synset]['connections'] = {}

            deepest_synset_dict[tmp_deepest_synset]['list_indx'].append(indx)

            for tmp_other_synset in tmp_deepest_synsets:
                if tmp_deepest_synset==tmp_other_synset:
                    continue

                if tmp_other_synset not in deepest_synset_dict[tmp_deepest_synset]['connections']:
                    deepest_synset_dict[tmp_deepest_synset]['connections'][tmp_other_synset] = 0
                deepest_synset_dict[tmp_deepest_synset]['connections'][tmp_other_synset] = deepest_synset_dict[tmp_deepest_synset]['connections'][tmp_other_synset] + 1

            if tmp_deepest_synset not in shapenet_synset_dict[item['shapenet_synset']]['listname']:
                shapenet_synset_dict[item['shapenet_synset']]['listname'].append(tmp_deepest_synset)

        '''
        sort_indx = [i[0] for i in sorted(enumerate(tmp_depth_list), key=lambda x:x[1])]
        for depth_indx in xrange(len(sort_indx) - 1):
            if not tmp_depth_list[sort_indx[depth_indx]]+1==tmp_depth_list[sort_indx[depth_indx+1]]:
                print(tmp_depth_list, tmp_tree_synset)
                break
        '''


    #print([len(depth_delta_dict[i]) for i in depth_delta_dict])

    print(len(deepest_synset_dict))
    len_of_obj = [len(deepest_synset_dict[key_tmp]['list_indx']) for key_tmp in deepest_synset_dict]
    len_of_obj.sort()
    print(len_of_obj)

    final_len = len(shapenet_synset_dict)
    final_synset = []
    for big_key in shapenet_synset_dict:
        print big_key, wn._synset_from_pos_and_offset('n',int(big_key[1:])),
        curr_size = len(shapenet_synset_dict[big_key]['listindx'])
        size_list = []
        print curr_size,
        for small_key in shapenet_synset_dict[big_key]['listname']:
            size_list.append((small_key, len(deepest_synset_dict[small_key]['list_indx'])))
            print "%s %i" % size_list[-1],
        print

        if big_key not in deepest_synset_dict:
            print('Base cate not as deepest!')
        if curr_size > args.minobjnum:
            tmp_larger_sons = [i[0] for i in size_list if i[1]>args.minobjnum]
            curr_split_len = len(tmp_larger_sons)
            final_len = final_len + curr_split_len -1
            final_synset.extend(tmp_larger_sons)
        else:
            final_synset.extend([big_key])

    #print(final_len)
    #print(len(final_synset))
    final_synset = np.unique(final_synset)
    obj_num = len(final_synset)
    #print(len(np.unique(final_synset)))

    len_of_obj  = [len(deepest_synset_dict[key_tmp]['list_indx']) for key_tmp in final_synset]
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

    for tmp_synset in final_synset:
        print tmp_synset, wn._synset_from_pos_and_offset('n',int(tmp_synset[1:])), len(deepest_synset_dict[tmp_synset]['list_indx']),
        for tmp_other_synset in deepest_synset_dict[tmp_synset]['connections']:
            if tmp_other_synset in final_synset:
                print tmp_other_synset, "%i/%i" % (deepest_synset_dict[tmp_synset]['connections'][tmp_other_synset], len(deepest_synset_dict[tmp_other_synset]['list_indx'])),
        print

'''
delete n04225987, n04524313, n03133538
combine n04502670, n04349401, n04599124
combine n04576211, n04225987, n04524313
combine n02831724, n02920259, n02920083
combine n03595860, n02690373
combine n03676759, n03085219, n03085602
'''
