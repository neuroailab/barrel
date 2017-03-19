import os

#dirload = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/ShapeNetCore.v2"
dirload = "/om/user/chengxuz/threedworld_related/shapenet_onlyobj/after_vhacd"

fin = open('obj_choice.txt', 'r')
lines = fin.readlines()

for line in lines:
    split_line = line.split(' ')
    dir_name = split_line[0][1:]
    subdir_name = split_line[1][:-1]

    new_obj_path = os.path.join(dirload, dir_name, subdir_name, subdir_name + '.obj')

    if not os.path.isfile(new_obj_path):
        print(line)
