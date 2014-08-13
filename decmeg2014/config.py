vw_path = '/Users/rodion/Documents/python/ml/bin/vw'
folder = '/home/rodion/decmeg2014/'
#folder = '/Users/rodion/decmeg2014/'

trainfiles = [
    'train_subject01.mat',
    'train_subject02.mat',
    'train_subject03.mat',
    'train_subject04.mat',
    'train_subject05.mat',
    'train_subject06.mat',
    'train_subject07.mat',
    'train_subject08.mat',
    'train_subject09.mat',
    'train_subject10.mat',
    'train_subject11.mat',
    'train_subject12.mat',
    'train_subject13.mat',
    'train_subject14.mat',
    'train_subject15.mat',
    'train_subject16.mat'
]

testfiles = [
    'test_subject17.mat',
    'test_subject18.mat',
    'test_subject19.mat',
    'test_subject20.mat',
    'test_subject21.mat',
    'test_subject22.mat',
    'test_subject23.mat'
]

model_folder = folder + 'model/'
model_logreg_folder = model_folder + 'logreg'

train_folder = folder + 'traindata/'
train_logreg_folder = train_folder + 'logreg'
train_spat_folder = train_folder + 'spat'
train_ica_folder = train_folder + 'fourierica/'
train_napca_folder = train_folder + 'napca/'

test_folder = folder + 'testdata/'
test_logreg_folder = test_folder + 'logreg'
test_spat_folder = test_folder + 'spat'
test_ica_folder = test_folder + 'fourierica/'
test_napca_folder = test_folder + 'napca/'

result_folder = folder + 'result/'

