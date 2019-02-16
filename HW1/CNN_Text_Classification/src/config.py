"""
This is the configuration file, change any parameters here
"""
train_file_path = '../topicclass/topicclass_train.txt'
dev_file_path = '../topicclass/topicclass_valid.txt'
test_file_path = '../topicclass/topicclass_test.txt'
test_result_path = '../topicclass/topicclass_result_lr04_train'
save_best = '../model04_t+d.pt'
pte_path = '../topicclass/GoogleNews-vectors-negative300.txt'
#pte_path = None
# number of k folds
k = 10
emb_dim = 300
# cnn feature map
filters = 100
batch_size = 128
lr = 1e-4
dropout = 0.5
# cnn kernel number
kernel_sizes = [3, 4, 5]
max_epochs = 20
most_frequent_pte = 300000
# random seed
random_seed = 947324
