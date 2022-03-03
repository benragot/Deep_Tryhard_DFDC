import os
import random

def move_test_files(data_file,path = None, fake = 'fake',real = 'real', x_fakes = 0.2, x_reals = 0.2):
    '''
    Function to split our train data into x% of test data.

    data_file : name of the data_train folder
    path : path to folder {data_file}, None when calling the function at the root
    fake : the name of the fake subfolder
    real : the name of the real subfolder
    x_fakes / x_reals : ratio of fakes and ratio of reals in case we want more fakes

    8==============D

    '''
    assert(data_file in os.listdir(path))
    os.mkdir(data_file + '_test') # Please don't end train folder name with train or TV
    fakes_test_path = os.path.join(data_file + '_test','fake_test')
    reals_test_path = os.path.join(data_file + '_test','real_test')
    os.makedirs(fakes_test_path)
    os.makedirs(reals_test_path)
    fakes_path = os.path.join(data_file,fake)
    reals_path = os.path.join(data_file,real)
    fake_list = os.listdir(fakes_path)
    real_list = os.listdir(reals_path)

    for i in range(int(len(fake_list)*x_fakes)):
        ffile = fake_list.pop(random.randint(0,len(fake_list)-i))
        os.replace(os.path.join(fakes_path,ffile),os.path.join(fakes_test_path,ffile))

    for i in range(int(len(real_list)*x_reals)):

        rfile = real_list.pop(random.randint(0,len(real_list)-i))
        os.replace(os.path.join(reals_path,rfile),os.path.join(reals_test_path,rfile))
