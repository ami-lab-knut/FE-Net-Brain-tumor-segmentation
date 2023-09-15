import numpy as np
import torch

from models import UNet, Discriminator, Decoder_2D, Decoder_3D, UNet_2D_red, UNet_3D_red, Decoder_3D_red, Dense_encoder_2d, Dense_encoder_3d, Dense_Decoder_2D, nnUNet_2D
from dataset import Prepare_dataset, Prepare_test_dataset, Prepare_full_volume_dataset, Data_slicer, TwoStreamBatchSampler
#from core import Run_Model
from core_full_volume import Run_Model

import random
from sklearn.model_selection import KFold

def set_divisions(selected, total_data, f):
    training = []
    validation = []
    testing = []
    prev = [6, 16, 20, 1, 23, 24]
    
    while True:
        rint = random.randint(0, 5)
        sb = f#prev[f] #rint
        if sb not in selected:
            selected.append(sb)
            break
    
    for sf in range(5):
        if sf == sb:
            testing = total_data[sf*57 : sf*57 + 57]
            validation = total_data[sf*57 : sf*57 + 57]
        else:
            k = total_data[sf*57:(sf+1)*57]
            for s in k:
                a1, a2, a3, a4, a5 = s
                training.append([a1, a2, a3, a4, a5])
            #training.append([total_data[sf*35:(sf+1)*35]])
    
    # k = total_data[1225:1251]
    # for s in k:
    #     a1, a2, a3, a4, a5 = s
    #     training.append([a1, a2, a3, a4, a5])
    training = np.array(training)
    
    return training, validation, testing, selected, sb

def combine_datasets(brats_18, brats_19, brats_20):
    combined_data = []
    
    for sample in brats_18:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    for sample in brats_19:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    for sample in brats_20:
        a1, a2, a3, a4, a5 = sample
        combined_data.append([a1, a2, a3, a4, a5])
        
    return np.array(combined_data)

# def set_divisions(selected, total_data, fold):
#     training = []
#     validation = []
#     testing = []
#     prev_inds = [6, 16, 20, 1, 23, 24]
#     while True:
#         rint = random.randint(0, 8) # 0 - 35
#         sb = 6 #prev_inds[fold]
#         if sb not in selected:
#             selected.append(sb)
#             break
    
#     for sf in range(35): # 35
#         if sf == sb:
#             testing = total_data[sf*35 : sf*35 + 20]
#             validation = total_data[sf*35 + 20 : (sf+1)*35]
#         else:
#             k = total_data[sf*35:(sf+1)*35]
#             for s in k:
#                 a1, a2, a3, a4, a5 = s
#                 training.append([a1, a2, a3, a4, a5])
#             #training.append([total_data[sf*35:(sf+1)*35]])
    
#     k = total_data[1225:1251]
#     # k = total_data[280:285]
#     for s in k:
#         a1, a2, a3, a4, a5 = s
#         training.append([a1, a2, a3, a4, a5])
#     training = np.array(training)
    
#     return training, validation, testing, selected, sb

# def set_divisions(selected, total_data, fold):
#     training = []
#     validation = []
#     testing = []
#     # prev_inds = [6, 16, 20, 1, 23, 24]
#     # while True:
#     #     rint = random.randint(0, 8) # 0 - 35
#     #     sb = 6 #prev_inds[fold]
#     #     if sb not in selected:
#     #         selected.append(sb)
#     #         break
#     sb = fold
#     for sf in range(5): # 35
#         if sf == sb:
#             testing = total_data[sf*250 : (sf+1)*250]
#             validation = total_data[sf*250 : (sf+1)*250]
#         else:
#             k = total_data[sf*250:(sf+1)*250]
#             for s in k:
#                 a1, a2, a3, a4, a5 = s
#                 training.append([a1, a2, a3, a4, a5])
#             #training.append([total_data[sf*35:(sf+1)*35]])
    
#     # k = total_data[1225:1251]
#     # k = total_data[280:285]
#     for s in k:
#         a1, a2, a3, a4, a5 = s
#         training.append([a1, a2, a3, a4, a5])
#     training = np.array(training)
    
#     return training, validation, testing, selected, sb

batch = 8
epochs = 100
base_lr = 0.001

data_path = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\Brain_data_paths_array.npy'
# data_path = 'D:\\brain_tumor_segmentation\\rough_4_modified_3d\\Brain_data_paths_array.npy'
total_data_20 = np.load(data_path, allow_pickle = True)

# data_path = 'D:\\brain_tumor_segmentation\\rough_4\\Brain_data_paths_array_2018.npy'
# total_data = np.load(data_path, allow_pickle = True)
# print('total data shape: ', total_data.shape)
# np.random.shuffle(total_data)
# np.save('shuffled_2018.npy', total_data)

total_data_18 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2018_x.npy', allow_pickle = True)
total_data_19 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2019_1.npy', allow_pickle = True)
total_data_20 = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2020.npy', allow_pickle = True)
total_data = np.load('D:\\brain_tumor_segmentation\\rough_4_modified_3d\\shuffled_2021.npy', allow_pickle = True)

combined_data = combine_datasets(total_data_18, total_data_19, total_data_20)
print('Combined data shape: ', combined_data.shape)


device = torch.device('cuda')

encoder_2d = nnUNet_2D(in_channels = 4, num_classes = 4, inter_channels = 32)#.cuda()
encoder_3d = nnUNet_2D(in_channels = 4, num_classes = 4, inter_channels = 32)#.cuda()
discriminator_1 = Discriminator(1024, 2, mode = '2D')#.cuda()
discriminator_2 = Discriminator(1024, 2, mode = '3D')#.cuda()
decoder = Decoder_2D(512, 32, 4)#.cuda() #2048
unet_3D = nnUNet_2D(in_channels = 4, num_classes = 4, inter_channels = 32)

k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)

selected = []

if __name__ == '__main__':
    for fold in range(1):
        
        train_data, validation_data, testing_data, selected, sb = set_divisions(selected, total_data, fold)
        
        print('train shape: ', train_data.shape)
        print('val shape: ', validation_data.shape)
        # train_data = train_data[0:50]
        validation_data = validation_data[0:15]
        testing_data = testing_data[0:2]
        combined_data = combined_data[0:15]
        
        ''' Train data prep '''
        train_slicer = Data_slicer(combined_data, slices = 7)
        primary_indices, secondary_indices = train_slicer.get_inds()
        
        train_sampler = TwoStreamBatchSampler(primary_indices, secondary_indices, batch, 1)
        train_set = Prepare_dataset(train_data, train_slicer.get_data(), slices = 7)
        
        Train_loader = torch.utils.data.DataLoader(train_set, num_workers = 3, pin_memory = True, batch_sampler = train_sampler)
        
        '''validation data prep '''
        validation_slicer = Data_slicer(validation_data, slices = 7)
        val_primary_indices, val_secondary_indices = validation_slicer.get_inds()
        
        validation_sampler = TwoStreamBatchSampler(val_primary_indices, val_secondary_indices, batch, 1)
        validation_set = Prepare_dataset(validation_data, validation_slicer.get_data(), slices = 7)
        
        Validation_loader = torch.utils.data.DataLoader(validation_set, num_workers = 3, pin_memory = True, batch_sampler = validation_sampler)
        
        
        weight_save_path = ['D:\\brain_tumor_segmentation\\weight_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\initial_training',
                            'D:\\brain_tumor_segmentation\\weight_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\regularization',
                            'D:\\brain_tumor_segmentation\\weight_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\combined',
                            'D:\\brain_tumor_segmentation\\weight_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\crf']
        
        record_save_path = ['D:\\brain_tumor_segmentation\\record_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\initial_training.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\regularization.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\combined.txt',
                            'D:\\brain_tumor_segmentation\\record_saves\\pretrain_nnUNet_only\\fold' + str(fold) +'\\testing.txt']
        
        
        trainer = Run_Model(weight_save_path, record_save_path, encoder_2d, encoder_3d, decoder, unet_3D, discriminator_1, discriminator_2)
        
        # test_set = Prepare_test_dataset(testing_data)
        # Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
        
        trainer.train_loop_mixed(30, base_lr, Train_loader, Validation_loader)#, sb)
        # trainer.Regularization_Loop(5, base_lr, Train_loader, Validation_loader)
        # trainer.Combined_loop(20, base_lr, Train_loader, Validation_loader, sb)
        
        # test_set = Prepare_test_dataset(testing_data)
        # Test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, pin_memory = True)
        # trainer.testing_whole_samples(Test_loader, 7, sb)
    

