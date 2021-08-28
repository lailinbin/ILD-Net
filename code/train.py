import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Unet, my_test_loss
from collections import defaultdict
import time
import copy
from torch import optim
from torch.optim import lr_scheduler
from data import Dataset2D


###########################################准备数据集#####################################################################


# 编写用于测试的简易测试集
class SimpleDataset(Dataset):

    def __init__(self, num: 'int' = 100, transform: 'function' = None) -> None:
        super().__init__()
        # self.data,self.label = torch.random((num,1,512,512))
        self.data = [np.random.rand(1, 512, 512)] * num
        self.label = [np.random.rand(1, 512, 512)] * num
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: 'int') -> list:
        data = self.data[item]
        label = self.label[item]
        if self.transform is not None:
            data = self.transform(data)
            label = self.transform(label)

        return data, label


########################################定义训练过程######################################################################


def train_model(
        model: 'torch.nn.Module',
        optimizer: 'torch.optim.optimizer',
        scheduler: 'torch.optim.lr_scheduler',
        data_loader: 'torch.utils.data.DataLoader',
        num_epochs: 'int' = 25,
        best_loss: 'int' = 1e10,
):
    """
    description: 定义模型训练过程
    optimizer: 训练优化器，加速训练过程
    scheduler: 用于根据epoch调整learning rate
    num_epochs: 迭代次数
    best_loss: 当损失函数达到best_loss时，即使epoch没有进行完成也退出训练过程
    """
    # 先保存模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        # 打印epoch信息和分割线
        print('_' * 200)
        print(f'Epoch {epoch + 1}/{num_epochs}')

        start = time.time()
        metrics = defaultdict(float)
        model.train()
        epoch_samples = 0

        # 打印optimizer的参数
        for param_group in optimizer.param_groups:
            print('LR', param_group['lr'])

        for inputs, labels in data_loader:
            # 将数据和标签移送入GPU如果有的话
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 将梯度参数置零
            optimizer.zero_grad()

            # 前向传播并最终训练日志
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = my_test_loss(outputs, labels, metrics)

                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()

            epoch_samples += inputs.size(0)

        # 计算一个epoch的平均损失函数
        epoch_loss = metrics['loss'] / epoch_samples

        # 如果平均损失比上一个epoch低则保存最新模型
        if epoch_loss < best_loss:
            print('saving best model')
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            print('epoch_loss has not been decreased! best_loss:{:.0f},epoch_loss:{:.0f}'.format(best_loss, epoch_loss))
        print_metrics(metrics, epoch_samples, 'train')

        time_elapsed = time.time() - start
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('eval_time:{:.0f}m{:.0f}s'.format(time_elapsed * (num_epochs - epoch) // 60,
                                                time_elapsed * (num_epochs - epoch) % 60))

    # 加载最好的模型
    model.load_state_dict(best_model_wts)
    return model


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append('{}:{:4f}'.format(k, metrics[k] / epoch_samples))
    print('{}:\n{}'.format(phase, ','.join(outputs)))


if __name__ == '__main__':

    # 初始化数据集
    train_set = SimpleDataset(10)
    eval_set = SimpleDataset(300)
    batch_size = 1

    ild_dataset = {
        'train_set': train_set,
        'eval_set': eval_set
    }

    ild_dataloader = {
        'train': DataLoader(train_set, batch_size, shuffle=True),
        'eval': DataLoader(eval_set, batch_size, shuffle=True)
    }

    ild_dataloader1 = DataLoader(train_set, batch_size=24, shuffle=True)

    """
    dataset_path = '.././2Dimg_data'
    ild_dataset = Dataset2D(dataset_path, label_index=4)
    ild_dataloader = DataLoader(dataset=ild_dataset, batch_size=1, shuffle=True)
    """
    # 训练模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('you are using {} for training'.format(device))

    num_class = 1
    model = Unet(num_class).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    model = train_model(model, optimizer, exp_lr_scheduler, data_loader=ild_dataloader["train"], num_epochs=5)
    torch.save(model.state_dict(), '.././experiment/model1.pkl')
    print('model has been saved!')
