from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False,TO_FILTER=True, f_block=None,device='cpu', crop_num=0, grayscale=False):
    since = time.time()

    val_acc_history = []
    loss_history = [[],[]]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                if grayscale == True:
                    inputs = np.repeat(inputs, 3, axis=1)  # duplicate grayscale image to 3 channels
                    
                inputs = inputs.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                if (TO_FILTER):
                  inputs = f_block(inputs,0)

                if crop_num > 0:
                    BS, CH, H, W = inputs.shape
                    inputs = inputs[:, :, crop_num:H - crop_num, crop_num:W - crop_num]

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                # scheduler.step()
                loss_history[0].append(epoch_loss)
            elif phase == 'val':
                loss_history[1].append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_history





# for feature_extract in [True,False]:
#     for TO_FILTER in [False,True]:
#
#         model_ft, input_size = D_Models.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#         # model_ft.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))  #Input grayscale
#
#         # Send the model to GPU
#         model_ft = model_non_filtered.to(device)
#
#         # Gather the parameters to be optimized/updated in this run. If we are
#         #  finetuning we will be updating all parameters. However, if we are
#         #  doing feature extract method, we will only update the parameters
#         #  that we have just initialized, i.e. the parameters with requires_grad
#         #  is True.
#         params_to_update_ = model_ft.parameters()
#         print("Params to learn:")
#         if feature_extract:
#             params_to_update = []
#             for name, param in model_ft.named_parameters():
#                 if param.requires_grad == True:
#                     params_to_update.append(param)
#                     print("\t", name)
#         else:
#             for name, param in model_ft.named_parameters():
#                 if param.requires_grad == True:
#                     print("\t", name)
#
#         # Observe that all parameters are being optimized
#         optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#
#         # Setup the loss fxn
#         criterion = nn.CrossEntropyLoss()
#
#         # Train and evaluate
#         model_ft, hist = Training_loop.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
#                                                    num_epochs=num_epochs, is_inception=(model_name == "inception"),
#                                                    device=device, TO_FILTER=TO_FILTER, f_block=F_B)
#
#         if (!feature_extract and !TO_FILTER):
#             nfe_nf_hist = [h.cpu().numpy() for h in hist]
#         elif (feature_extract and !TO_FILTER):
#             fe_nf_hist = [h.cpu().numpy() for h in hist]
#         elif (!feature_extract and TO_FILTER):
#             nfe_f_hist = [h.cpu().numpy() for h in hist]
#         elif (feature_extract and TO_FILTER):
#             fe_f_hist = [h.cpu().numpy() for h in hist]