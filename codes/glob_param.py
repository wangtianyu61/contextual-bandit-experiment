choice = 'loan'
if choice == 'openml':
    DS_DIR = '../datasets/vwdatasets/'
    DS_SPLIT = '../datasets/vwdatasets/train_test_split/'
    DS_RES = '../res/FALCON_ALL/FALCON_vw/'
    DS_RESULT = '../res/train_test_split/vw/'

elif choice == 'hotel':
    DS_DIR = '../datasets/pricing/MSOM_Hotel/'
    DS_RES = '../res/FALCON_ALL/FALCON_HOTEL/'
    DS_SPLIT = '../datasets/pricing/MSOM_Hotel/train_test_split/'
    DS_RESULT = '../res/train_test_split/hotel/'

elif choice == 'MS':
    DS_DIR = '../datasets/MSLR-WEB10K/'
    DS_RES = '../res/FALCON_ALL/FALCON_ms/'
    DS_SPLIT = '../datasets/MSLR-WEB10K/train_test_split/'
    DS_RESULT = '../res/train_test_split/ms/'
    
elif choice == 'yahoo':
    DS_DIR = '../datasets/yahoo_dataset/'
    DS_RES = '../res/FALCON_ALL/FALCON_yahoo/'
    DS_SPLIT = '../datasets/yahoo_dataset/train_test_split/'
    DS_RESULT = '../res/train_test_split/yahoo/'
    
elif choice == 'loan':
    DS_DIR = '../datasets/OnlineAutoLoan/'
    DS_RES = '../res/FALCON_ALL/FALCON_loan/'
    DS_SPLIT = '../datasets/OnlineAutoLoan/train_test_split/'
    DS_RESULT = '../res/train_test_split/loan/'
    