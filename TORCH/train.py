import time
import torch
import copy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

def train_model_logits(num_epochs, model, dataloader, classes_labels, criterion, optimizer, scheduler, device, save, save_name ):
    """
    
    Preparado para trabajar directamente con los logits
    
    """

    writer = SummaryWriter()  #launch tensorboard

    since = time.time()

    torch.cuda.empty_cache()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    acc_list_train = []              #listas para almacenar resultados
    loss_list_train = []
    acc_list_val = []
    loss_list_val = []
    normal = torch.nn.Softmax(dim=1)

    dataset_sizes = {'train': len(dataloader['train'].dataset), 'val': len(dataloader['val'].dataset)}
    #invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'EPOCH {epoch}/{num_epochs - 1}')
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f'Starting {phase} phase...')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_target = np.array([])
            y_pred = np.array([])
            y_outputs = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloader[phase]):
                #image_tb = inputs
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #target_val = labels.cpu().numpy()
                        #for idx,v in enumerate(target_val):
                        #    img_save = invTrans(image_tb[idx])
                        #    writer.add_image("IMG ERROR: "+str(idx),img_save,0)
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        y_target = np.concatenate((y_target,labels.cpu().numpy()))
                        y_pred = np.concatenate((y_pred,preds.detach().cpu().numpy()))
                        for probs in normal(outputs).detach().cpu().numpy():
                            y_outputs.append(probs)
                        """target_val = labels.cpu().numpy()
                        pred_val = preds.detach().cpu().numpy()
                        for idx,v in enumerate(target_val):
                            if v != pred_val[idx]:
                                img_save = invTrans(image_tb[idx])
                                #data_transforms = transforms.toPIL
                                #img = data_transforms(transforms.ToPIL)
                                #writer.add_image("IMG ERROR: "+str(idx),img_save,0)
                                writer.add_image("REAL: "+str(classes_labels[v])+" vs PRED: "+str(classes_labels[pred_val[idx]]),img_save,0)
                        y_target = np.concatenate((y_target,target_val))
                        y_pred = np.concatenate((y_pred,pred_val))"""

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.cpu() == labels.data.cpu())
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                scheduler.step()
                acc_list_train.append(epoch_acc)
                loss_list_train.append(epoch_loss)
                writer.add_scalar('Acc/train', epoch_acc, epoch )
                writer.add_scalar('Loss/train', epoch_loss, epoch )
            
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val':
                #print(y_target.flatten(),"\n",y_pred.flatten())
                acc_list_val.append(epoch_acc)
                loss_list_val.append(epoch_loss)
                writer.add_scalar('Acc/val', epoch_acc, epoch )
                writer.add_scalar('Loss/val', epoch_loss, epoch )
                report = classification_report(y_target.flatten(),y_pred.flatten(), target_names=classes_labels)
                confusion_m = confusion_matrix(y_target.flatten(),y_pred.flatten())
                #disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m, display_labels=classes_labels)
                fig = display_cm(confusion_m, classes_labels)
                writer.add_figure('Confusion Matrix', fig, epoch)
                print("Report:\n", report)#, "\n", "CM:", confusion_m)
                #roc_auc = roc_auc_score(y_target, np.array(y_outputs), multi_class='ovr')
                #writer.add_scalar('ROC_AUC', roc_auc, epoch )
                #print("ROC_AUC:", roc_auc)

                # plot all the pr curves
                for i in range(len(classes_labels)):
                    add_pr_curve_tensorboard(i, np.array(y_outputs), y_target, classes_labels, writer)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    #if save is true, it saves the best model
    if save:

        torch.save(model.state_dict(), save_name)

    return model, acc_list_train, loss_list_train, acc_list_val, loss_list_val

def display_cm(cm, labels):
    """
    Funcion mostrar matriz confusion formato para TensorBoard
    """

    fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    tick_marks = np.arange(len(labels))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    fig.colorbar(im)

    for i, j in zip(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig


def add_pr_curve_tensorboard(class_index, test_probs, test_label, classes, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


""" 
def train_model(model, dataloader, criterion, optimizer, scheduler, device, save, classes_labels, num_epochs=25):
    ""
    
    Preparado para usar labels en formato one-hot encoding, se necesita usar nn.Softmax() en la red
    
    ""
    since = time.time()

    torch.cuda.empty_cache()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    acc_list = []
    loss_list = []


    dataset_sizes = {'train': len(dataloader['train'].dataset), 'val': len(dataloader['val'].dataset)}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_target = np.array([])
            y_pred = np.array([])
            y_outputs = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloader[phase]):
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                   
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    
                    real_pred = torch.argmax(outputs, dim=1)
                    real_targ = torch.argmax(labels.data, dim=1)
                    #print("output:",outputs,"\nlabels:", labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        y_target = np.concatenate((y_target,real_targ.cpu().numpy()))
                        y_pred = np.concatenate((y_pred,real_pred.detach().cpu().numpy()))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(real_pred == real_targ)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(type(epoch) , type(epoch_loss))
            acc_list.append(epoch_acc)
            loss_list.append(epoch_loss)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val':
                #print(y_target.flatten(),"\n",y_pred.flatten())
                report = classification_report(y_target.flatten(),y_pred.flatten(), labels=classes_labels)
                confusion_m = confusion_matrix(y_target.flatten(),y_pred.flatten(), labels=classes_labels)
                print("Report:\n", report, "\n", "CM:", confusion_m)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    #if save is true, it saves the best model
    if save:

        torch.save(model.state_dict(), "model.pth")

    return model, acc_list.cpu(), loss_list.cpu()
"""