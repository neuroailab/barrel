import os

def get_file_list():
    file_folder = '/mnt/fs0/chengxuz/Data/nd_response'

    subfolder_list = ['spatemp_sm2_RDMs_obj', 'temp_spa_RDMs_s0_obj', 'temp_spa_RDMs_s1_obj',
           'temp_spa_RDMs_s2_obj', 'temp_spa_RDMs_s3_obj', 'temp_spa_RDMs_s4_obj', 'fdb_RDMs_obj', 'spa_temp_RDMs_obj']

    file_list = []

    for now_folder in subfolder_list:
        file_list.extend([os.path.join(file_folder, now_folder, v) for v in os.listdir(os.path.join(file_folder, now_folder))])

    print(len(file_list))
    file_list = sorted(file_list, key = lambda x: os.path.getmtime(x))
    #print(file_list)
    return file_list

if __name__ == '__main__':
    #main()
    file_list = get_file_list()
    print(len(file_list))
