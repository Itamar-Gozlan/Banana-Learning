import os

'''
this script will fix the naming on the thermal images that for some unknown reason
starts with 3 non conventional names ("xxx_same_conditions") and count from  1 
'''


def contains(list1, filter1):
    for x in list1:
        if filter1(x):
            return True
    return False


def fix_dir_names(dirnames):
    sorted_dirs = sorted(dirnames)
    new_list = []
    for i in range(len(sorted_dirs)):
        if sorted_dirs[i].find('_same_') >= 0:
            new_list.append(sorted_dirs[i][0:sorted_dirs[i].find('_same_')] + '_day_' + str(i + 1))
        elif sorted_dirs[i].find('_day_') >= 0:
            new_list.append(sorted_dirs[i][0:sorted_dirs[i].find('_day_')] + '_day_' + str(i+1))
        else:
            continue
    return new_list


def rename_directories(path, dirnames):
    sorted_dirs = sorted(dirnames)
    new_dir_names = fix_dir_names(sorted_dirs)
    for i in range(len(sorted_dirs)):
        src = os.path.join(path, sorted_dirs[i])
        dst = os.path.join(path, new_dir_names[i])
        os.rename(src, dst)


def foldername_fix(path, dirnames):
    if contains(dirnames, lambda x: x.find('same_condition') >=0):
        rename_directories(path, dirnames)


def fix_names(path):
    for dirpath, dirnames, filenames in os.walk(path):
        foldername_fix(dirpath, dirnames)


def main():
    # TODO: verify path before running
    PATH = '/run/media/eyal/Elements/banana_ml/thermal/'  # this is the path to my thermal images
    fix_names(PATH)


if __name__ == '__main__':
    main()