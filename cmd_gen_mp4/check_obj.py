import os

def get_list(fin_name, dirload):
    fin = open(fin_name, 'r')
    lines = fin.readlines()

    ret_list = []

    for line in lines:
        split_line = line.split(' ')
        dir_name = split_line[0][1:]
        subdir_name = split_line[1][:-1]

        new_obj_path = os.path.join(dirload, dir_name, subdir_name, subdir_name + '.obj')
        ret_list.append((subdir_name, new_obj_path))

    return ret_list

obj_list_fname = 'obj_choice_2.txt'
#load_path0 = '/om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd'
#load_path1 = '/om/user/chengxuz/threedworld_related/shapenet_onlyobj/ShapeNetCore.v2'
load_path0 = '/scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/after_vhacd'
load_path1 = '/scratch/users/chengxuz/barrel/barrel_relat_files/all_objs/shapenet_onlyobj/ShapeNetCore.v2'

list_0 = get_list(obj_list_fname, load_path0)
list_1 = get_list(obj_list_fname, load_path1)

num_bad = 0

for obj_indx in xrange(len(list_0)):
    size_0 = os.path.getsize(list_0[obj_indx][1])
    size_1 = os.path.getsize(list_1[obj_indx][1])
    if size_1 > 50*size_0:
        num_bad = num_bad+1

    if obj_indx%100==0:
        print("Indx %i, %i" % (obj_indx, num_bad))

print(num_bad)
