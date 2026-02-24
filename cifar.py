import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import  torch.utils.data as Data
from cifar_n.cifarn import CIFAR100N
def get_fast(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    t_num=2
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar_/'):
        os.makedirs('./data/binary_cifar_')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar_'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar_'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        # print("T",t)
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n
    Loder={}
    Loder_test={}
    for t in range(5):
        # print("t",t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        Loder_test[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        u2 = torch.tensor(data[t]['test']['x'].reshape(-1, 3, 32, 32))
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        #u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            # batch_size=10,
            batch_size=10,
            shuffle=True,
            )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
        #Loder[t]['valid'] = valid_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_dataset= datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test_dataset=datasets.CIFAR10('./data/', train=False, download=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10//data[0]['ncla']], size,Loder,test_loader

def get_cifar_data(dataset_name, batch_size, n_workers):
    data = {}
    size = [3, 32, 32]
    if dataset_name == "cifar10":
        task_num = 5
        class_num = 10
        data_dir = './data/binary_cifar_/'
    elif dataset_name == "cifar100":
        task_num = 10
        class_num = 100
        data_dir = './data/binary_cifar100_10/'
    class_per_task = class_num // task_num

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = './data/'
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = {}
        if dataset_name == "cifar10":
            dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
            dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * task_id, class_per_task * (task_id + 1)):
                        data[task_id][data_type]['x'].append(image)
                        data[task_id][data_type]['y'].append(label)

        # save
        for task_id in data.keys():
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
                data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
                torch.save(data[task_id][data_type]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size

def get_cifar100_50(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_222/'):
        os.makedirs('./data/binary_cifar100_222')
        t_class_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for t in range(100 // t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_222'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_222'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(51))
    print('Task order =', ids)
    for i in range(51):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100_222'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100_222'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(50):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100_10(seed=0, pc_valid=0.10, batch=10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100_10/'):
        os.makedirs('./data/binary_cifar100_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for t in range(100 // t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100_10'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_mnist(seed=0,pc_valid=0.10):
    data = {}

    taskcla = []
    size = [1, 28, 28]
    # CIFAR10
    if not os.path.isdir('./data/binary_mnist_2/'):
        os.makedirs('./data/binary_mnist_2')
        t_class_num = 2
        mean = (0.1307,)
        std = (0.3081,)
        dat={}
        dat['train']=datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.MNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'mnist' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_class_num
        data[t] = {}
        data[t]['name'] = 'mnist-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_mnist_2'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_mnist_2'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist_2'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist_2'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'mnist->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(5):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 1, 28, 28))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = (0.1307,)
    std = (0.3081,)

    test_dataset = datasets.MNIST('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_mnist_fast(seed=0,pc_valid=0.10):
    data = {}

    taskcla = []
    size = [1, 28, 28]
    # CIFAR10
    if not os.path.isdir('./data/binary_mnist/'):
        os.makedirs('./data/binary_mnist')
        t_class_num = 2
        mean = (0.1307,)
        std = (0.3081,)
        dat={}
        dat['train']=datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.MNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'mnist' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_class_num
        data[t] = {}
        data[t]['name'] = 'mnist-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_mnist'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_mnist'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnist'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'mnist->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(5):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 1, 28, 28))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = (0.1307,)
    std = (0.3081,)

    test_dataset = datasets.MNIST('./data/', train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader


from tinyimagenet import MyTinyImagenet
from conf import base_path


def get_tinyimagenet_100_fast(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 64, 64]
    # CIFAR10
    if not os.path.isdir('./data/binary_tiny200_2/'):
        os.makedirs('./data/binary_tiny200_2')
        t_class_num = 2
        # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        # std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path() + 'TINYIMG',
                               train=True, download=True, transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyTinyImagenet(base_path() + 'TINYIMG',
                              train=False, download=True, transform=test_transform)
        dat[
            'train'] = train  # datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat[
            'test'] = test  # datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(200 // t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 200 // t_class_num
        data[t] = {}
        data[t]['name'] = 'tiny200-all'
        data[t]['ncla'] = 200
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_2'),
                                        'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_2'),
                                        'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_tiny200_2'),
                             'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_tiny200_2'),
                             'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(100):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 64, 64))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyTinyImagenet(base_path() + 'TINYIMG',
                          train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader


from tinyimagenet import MyTinyImagenet
from conf import base_path
def get_tinyimagenet_100(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 64, 64]
    # CIFAR10
    if not os.path.isdir('./data/binary_tiny200_22/'):
        os.makedirs('./data/binary_tiny200_22')
        t_class_num = 2
        #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        #std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path() + 'TINYIMG',
                               train=True, download=True, transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyTinyImagenet(base_path() + 'TINYIMG',
                              train=False, download=True, transform=test_transform)
        dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(200//t_class_num):
            # print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 200 // t_class_num
        data[t] = {}
        data[t]['name'] = 'tiny200-all'
        data[t]['ncla'] = 200
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_22'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_tiny200_22'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_tiny200_22'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_tiny200_22'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        # print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(100):
        # print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 64, 64))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyTinyImagenet(base_path() + 'TINYIMG',
                          train=False, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader

from imagenet32 import MyImagenet32
from conf import base_path


def get_img32_100(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_img32/'):
      
        os.makedirs('./data/binary_img32')
        t_class_num = 100
        #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        #std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyImagenet32(base_path() + 'IMG32',
                               train=True,  transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyImagenet32(base_path() + 'IMG32',
                              train=False, transform=test_transform)
        dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(1000//t_class_num):
            # print(t)
            data[t] = {}
            data[t]['name'] = 'img32-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num*t, t_class_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 1000 // t_class_num
        data[t] = {}
        data[t]['name'] = 'img32-1000-all'
        data[t]['ncla'] = 1000
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_img32'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_img32'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        # print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(100):
        # print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyImagenet32(base_path() + 'IMG32',
                          train=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader


def get_img32_100_part(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_img32_part50/'):
      
        os.makedirs('./data/binary_img32_part50')
        #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        #std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyImagenet32(base_path() + 'IMG32',
                               train=True,  transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyImagenet32(base_path() + 'IMG32',
                              train=False, transform=test_transform)
        dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        
        total_class_num = 1000
        t_class_num = 10
        label_dic = {}
        last_t = total_class_num // t_class_num
        data[last_t] = {}
        data[last_t]['name'] = 'img32-1000-all'
        data[last_t]['ncla'] = total_class_num
        data[last_t]['train'] = {'x': [], 'y': []}
        data[last_t]['test'] = {'x': [], 'y': []}        
        for t in range(1000//t_class_num):
            # print(t)
            data[t] = {}
            data[t]['name'] = 'img32-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                if s == 'train':
                    for image, target in loader:
                        label = target.numpy()[0]
                        if len(data[t][s]['x'])>=t_class_num * 50:
                            break
                        if label_dic.get(label) == None:
                            label_dic[label] = 0
                        elif label_dic[label]>=50:
                            continue

                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)
                            data[last_t][s]['x'].append(image)
                            data[last_t][s]['y'].append(label)
                            label_dic[label] = label_dic[label] + 1
                else:
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            data[last_t][s]['x'].append(image)
                            data[last_t][s]['y'].append(label)
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)

        # t = 1000 // t_class_num
        # data[t] = {}
        # data[t]['name'] = 'img32-1000-all'
        # data[t]['ncla'] = 1000
        # for s in ['train', 'test']:
        #     loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        #     data[t][s] = {'x': [], 'y': []}
        #     for image, target in loader:
        #         label = target.numpy()[0]
        #         data[t][s]['x'].append(image)
        #         data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_img32_part50'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_img32_part50'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(101))
    print('Task order =', ids)
    for i in range(101):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_part50'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_part50'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'img32->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        # print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(100):
        # print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyImagenet32(base_path() + 'IMG32',
                          train=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader

def get_img32_10(seed=0,pc_valid=0.10, ):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_img32_10/'):
      
        os.makedirs('./data/binary_img32_10')
        #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        #std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyImagenet32(base_path() + 'IMG32',
                               train=True,  transform=test_transform)
        # train = datasets.CIFAR100('Data/', train=True,  download=True)
        test = MyImagenet32(base_path() + 'IMG32',
                              train=False, transform=test_transform)
        dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

        total_class_num = 1000
        t_class_num = 100
        label_dic = {}
        last_t = total_class_num // t_class_num
        data[last_t] = {}
        data[last_t]['name'] = 'img32-1000-all'
        data[last_t]['ncla'] = total_class_num
        data[last_t]['train'] = {'x': [], 'y': []}
        data[last_t]['test'] = {'x': [], 'y': []}
        for t in range(total_class_num//t_class_num):
            # print(t)
            data[t] = {}
            data[t]['name'] = 'img32-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=True)
                data[t][s] = {'x': [], 'y': []}
                if s == 'train':
                    for image, target in loader:
                        label = target.numpy()[0]
                        if len(data[t][s]['x'])>=t_class_num * 200:
                            break
                        if label_dic.get(label) == None:
                            label_dic[label] = 0
                        elif label_dic[label]>=200:
                            continue

                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)
                            data[last_t][s]['x'].append(image)
                            data[last_t][s]['y'].append(label)
                            label_dic[label] = label_dic[label] + 1
                else:
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            data[last_t][s]['x'].append(image)
                            data[last_t][s]['y'].append(label)
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)

        # t = 1000 // t_class_num
        # data[t] = {}
        # data[t]['name'] = 'img32-1000-all'
        # data[t]['ncla'] = 1000
        # for s in ['train', 'test']:
        #     loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        #     data[t][s] = {'x': [], 'y': []}
        #     for image, target in loader:
        #         label = target.numpy()[0]
        #         data[t][s]['x'].append(image)
        #         data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_img32_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_img32_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_10'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_10'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'img32->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        # print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        # print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader

    transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                     (0.2770, 0.2691, 0.2821))
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transform])
    test = MyImagenet32(base_path() + 'IMG32',
                          train=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader

# def get_img32_10(seed=0,pc_valid=0.10, ):
#     data = {}
#     taskcla = []
#     size = [3, 32, 32]
#     # CIFAR10
#     if not os.path.isdir('./data/binary_img32_10/'):
      
#         os.makedirs('./data/binary_img32_10')
#         #mean = [x / 255 for x in [125.3, 123.0, 113.9]]
#         #std = [x / 255 for x in [63.0, 62.1, 66.7]]
#         dat={}
#         transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
#                                          (0.2770, 0.2691, 0.2821))

#         test_transform = transforms.Compose(
#             [transforms.ToTensor(), transform])

#         train = MyImagenet32(base_path() + 'IMG32',
#                                train=True,  transform=test_transform)
#         # train = datasets.CIFAR100('Data/', train=True,  download=True)
#         test = MyImagenet32(base_path() + 'IMG32',
#                               train=False, transform=test_transform)
#         dat['train']=train#datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
#         dat['test']=test #datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

#         total_class_num = 20
#         t_class_num = 2
#         label_dic = {}
#         choosen_cls = []
#         last_t = total_class_num // t_class_num
#         data[last_t] = {}
#         data[last_t]['name'] = 'img32-1000-all'
#         data[last_t]['ncla'] = total_class_num
#         data[last_t]['train'] = {'x': [], 'y': []}
#         data[last_t]['test'] = {'x': [], 'y': []}
#         for t in range(total_class_num//t_class_num):
#             # print(t)
#             data[t] = {}
#             data[t]['name'] = 'img32-' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
#             data[t]['ncla'] = t_class_num
#             for s in ['train', 'test']:
#                 loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=True)
#                 data[t][s] = {'x': [], 'y': []}
#                 if s == 'train':
#                     for image, target in loader:
#                         label = target.numpy()[0]
#                         if len(data[t][s]['x'])>=t_class_num * 1000:
#                             break

#                         # if len(choosen_cls) < (t + 1) * t_class_num:
#                         #     choosen_cls.append(label)
#                         # if label not in choosen_cls:
#                         #     continue

#                         if label_dic.get(label) == None:
#                             label_dic[label] = 0
#                         elif label_dic[label]>=1000:
#                             continue

#                         if label in range(t_class_num*t, t_class_num*(t+1)):
#                             data[t][s]['x'].append(image)
#                             data[t][s]['y'].append(label)
#                             data[last_t][s]['x'].append(image)
#                             data[last_t][s]['y'].append(label)
#                             label_dic[label] = label_dic[label] + 1
#                 else:
#                     for image, target in loader:
#                         label = target.numpy()[0]
#                         if label in range(t_class_num*t, t_class_num*(t+1)):

#                             data[last_t][s]['x'].append(image)
#                             data[last_t][s]['y'].append(label)
#                             data[t][s]['x'].append(image)
#                             data[t][s]['y'].append(label)

#         # t = 1000 // t_class_num
#         # data[t] = {}
#         # data[t]['name'] = 'img32-1000-all'
#         # data[t]['ncla'] = 1000
#         # for s in ['train', 'test']:
#         #     loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
#         #     data[t][s] = {'x': [], 'y': []}
#         #     for image, target in loader:
#         #         label = target.numpy()[0]
#         #         data[t][s]['x'].append(image)
#         #         data[t][s]['y'].append(label)

#         # "Unify" and save
#         for t in data.keys():
#             for s in ['train', 'test']:
#                 data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
#                 data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
#                 torch.save(data[t][s]['x'],
#                            os.path.join(os.path.expanduser('./data/binary_img32_10'), 'data' + str(t) + s + 'x.bin'))
#                 torch.save(data[t][s]['y'],
#                            os.path.join(os.path.expanduser('./data/binary_img32_10'), 'data' + str(t) + s + 'y.bin'))
#     # Load binary files
#     data = {}
#     ids = list(np.arange(11))
#     print('Task order =', ids)
#     for i in range(11):
#         data[i] = dict.fromkeys(['name','ncla','train','test'])
#         for s in ['train','test']:
#             data[i][s]={'x':[],'y':[]}
#             data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_10'),'data'+str(ids[i])+s+'x.bin'))
#             data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_img32_10'),'data'+str(ids[i])+s+'y.bin'))
#         data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
#         data[i]['name'] = 'img32->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

#     # Others
#     n = 0
#     for t in data.keys():
#         # print("T", t)
#         taskcla.append((t, data[t]['ncla']))
#         n += data[t]['ncla']
#     data['ncla'] = n
#     Loder = {}
#     for t in range(10):
#         # print("t", t)
#         Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
#         u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
#         # print("u1",u1.size())

#         TOTAL_NUM = u1.size()[0]
#         NUM_VALID = int(round(TOTAL_NUM * 0.1))
#         NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
#         # u1.size()[0]
#         # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
#         # u3 = data[t]['valid']['x']
#         # print("u3",u3.size(),s)
#         # u4=data[t]['valid']['y']
#         dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
#         dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
#         train_loader = torch.utils.data.DataLoader(
#             dataset_new_train,
#             batch_size=10,
#             shuffle=True,
#         )
#         test_loader = torch.utils.data.DataLoader(
#             dataset_new_test,
#             batch_size=64,
#             shuffle=True,
#         )
#         Loder[t]['train'] = train_loader
#         Loder[t]['test'] = test_loader

#     transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
#                                      (0.2770, 0.2691, 0.2821))
#     test_transform = transforms.Compose(
#         [transforms.ToTensor(), transform])
#     test = MyImagenet32(base_path() + 'IMG32',
#                           train=False, transform=test_transform)

#     test_loader = torch.utils.data.DataLoader(
#         test,
#         batch_size=64,
#         shuffle=True,
#     )
#     print("Loder is prepared")
#     return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader
def get_cifar100n_10(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    train_cifar100_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_cifar100_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if not os.path.isdir('./data/binary_cifar100n_10/'):
        os.makedirs('./data/binary_cifar100n_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = CIFAR100N(root='./data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type='noisy_label',
                                noise_path = './data/CIFAR-100_human.pt', is_human=True
                            )
        dat['test'] = CIFAR100N(root='./data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type='noisy_label',
                            )
        for t in range(100 // t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100n-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target, _ in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100n-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target, _ in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100n_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100n_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100n_10'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100n_10'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100n->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = CIFAR100N(root='./data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type='noisy_label',
                            )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader


import random
def get_lt_dict(img_max, cls_num, imb_factor):
    lt_dict = {}
    count = {}

    # 生成类别索引并打乱顺序
    cls_indices = list(range(cls_num))
    random.shuffle(cls_indices)  # 乱序类别索引

    # 生成长尾类别分布
    for rank, cls_idx in enumerate(cls_indices):        
        lt_dict[cls_idx] = img_max * (imb_factor**(rank / (cls_num - 1.0)))  # 按乱序的索引赋值
        count[cls_idx] = 0  # 初始化计数

    return lt_dict, count

def get_cifar100lt_10(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100lt_10/'):
        os.makedirs('./data/binary_cifar100lt_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        lt_dict, count = get_lt_dict(img_max=len(dat['train'])//100, cls_num=100, imb_factor=0.01)

        for t in range(100 // t_class_num):
            print(t)
            
            data[t] = {}
            data[t]['name'] = 'cifar100lt-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)) and count[label] < lt_dict[label]:
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
                        count[label] = count[label] + 1
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-lt'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100lt_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100lt_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100lt_10'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100lt_10'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader


def get_cifar100lt10_10(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100lt10_10/'):
        os.makedirs('./data/binary_cifar100lt10_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        lt_dict, count = get_lt_dict(img_max=len(dat['train'])//100, cls_num=100, imb_factor=0.1)

        for t in range(100 // t_class_num):
            print(t)
            
            data[t] = {}
            data[t]['name'] = 'cifar100lt-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)) and count[label] < lt_dict[label] and s == 'train':
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
                        count[label] = count[label] + 1
                    if label in range(t_class_num * t, t_class_num * (t + 1)) and s=='test':
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100-lt'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100lt10_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100lt10_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100lt10_10'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100lt10_10'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader

import torch.nn.functional as F
def transform_to_cifar_style(image):
    """
    将 1×28×28 的 MNIST 图片转换为 3×32×32 的 CIFAR 风格
    """
    flag=0
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
        flag=1
    image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)  # 调整大小
    image = image.repeat(1, 3, 1, 1)  # 复制通道，从 1×32×32 变成 3×32×32
    if flag:
        image = image.squeeze(0)
    return image
def get_mnistcifar_2(seed=0,pc_valid=0.10):
    data = {}

    taskcla = []
    size = [1, 28, 28]
    size2 = [3, 32, 32]
    if not os.path.isdir('./data/binary_mnistcifar_2/'):
        os.makedirs('./data/binary_mnistcifar_2')
        t_class_num = 2
        mean = (0.1307,)
        std = (0.3081,)
        dat={}
        dat['train']=datasets.MNIST('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.MNIST('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        
        mean2 = [x / 255 for x in [125.3, 123.0, 113.9]]
        std2 = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat2 = {}
        dat2['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean2,std2)]))
        dat2['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean2,std2)]))
        
        for t in range(20//t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'mnistcifar' + str(t_class_num*t) + '-' + str(t_class_num*(t+1)-1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                if t < 5:
                    loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                    data[t][s] = {'x': [], 'y': []}
                    for image, target in loader:
                        label = target.numpy()[0]
                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            image = transform_to_cifar_style(image)
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)
                else:
                    loader2 = torch.utils.data.DataLoader(dat2[s], batch_size=1, shuffle=False)
                    data[t][s] = {'x': [], 'y': []}
                    for image, target in loader2:
                        label = target.numpy()[0] + 10
                        if label in range(t_class_num*t, t_class_num*(t+1)):
                            data[t][s]['x'].append(image)
                            data[t][s]['y'].append(label)


        t = (10+10) // t_class_num
        data[t] = {}
        data[t]['name'] = 'mnistcifar-all'
        data[t]['ncla'] = (10+10)
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                image = transform_to_cifar_style(image)
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)
            loader2 = torch.utils.data.DataLoader(dat2[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader2:
                label = target.numpy()[0] + 10
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)
        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size2[0], size2[1], size2[2])
                # if t < 5:
                #     data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                # else:
                #     data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size2[0], size2[1], size2[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_mnistcifar_2'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_mnistcifar_2'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnistcifar_2'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_mnistcifar_2'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'mnistcifar->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        # u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 1, 28, 28))  # .item()
        # u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 1, 32, 32))
        # # print("u1",u1.size())

        # TOTAL_NUM = u1.size()[0]
        # NUM_VALID = int(round(TOTAL_NUM * 0.1))
        # NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=32,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = (0.1307,)
    std = (0.3081,)
    mean2 = [x / 255 for x in [125.3, 123.0, 113.9]]
    std2 = [x / 255 for x in [63.0, 62.1, 66.7]]
    class ModifiedCIFAR10(torch.utils.data.Dataset):
        def __init__(self, dataset, label_offset):
            self.dataset = dataset
            self.label_offset = label_offset

        def __getitem__(self, index):
            image, label = self.dataset[index]
            # 修改标签，将其从 [0, 9] 变为 [label_offset, label_offset+9]
            return image, label + self.label_offset

        def __len__(self):
            return len(self.dataset)
    class ModifiedMNist(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            image, label = self.dataset[index]
            image = transform_to_cifar_style(image)
            return image, label

        def __len__(self):
            return len(self.dataset)
    # MNIST 数据集
    test_dataset = datasets.MNIST('./data/', train=False, download=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    # CIFAR-10 数据集
    test_dataset2 = datasets.CIFAR10('./data/', train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean2, std2)]))

    # 修改 CIFAR-10 数据集的标签，使其从 [0, 9] 改为 [10, 19]
    modified_cifar10 = ModifiedCIFAR10(test_dataset2, label_offset=10)
    modified_mnist = ModifiedMNist(test_dataset)
    # 合并两个数据集
    combined_test_dataset = torch.utils.data.ConcatDataset([modified_mnist, modified_cifar10])

    # 创建 DataLoader
    test_loader = torch.utils.data.DataLoader(
        combined_test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:(10+10) // data[0]['ncla']], size, Loder, test_loader
def get_cifar100GSN_10(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar100GSN_10/'):
        os.makedirs('./data/binary_cifar100GSN_10')
        t_class_num = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat = {}
        dat['train'] = datasets.CIFAR100('./data/', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std), GaussianNoise(mean=0., min_std=0.05, max_std=0.2, probability=0.1),]))
        dat['test'] = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for t in range(100 // t_class_num):
            print(t)
            data[t] = {}
            data[t]['name'] = 'cifar100gsn-' + str(t_class_num * t) + '-' + str(t_class_num * (t + 1) - 1)
            data[t]['ncla'] = t_class_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_class_num * t, t_class_num * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 100 // t_class_num
        data[t] = {}
        data[t]['name'] = 'cifar100gsn-all'
        data[t]['ncla'] = 100
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100GSN_10'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar100GSN_10'), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(11))
    print('Task order =', ids)
    for i in range(11):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100GSN_10'), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser('./data/binary_cifar100GSN_10'), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar100gsn->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n = 0
    for t in data.keys():
        print("T", t)
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    Loder = {}
    for t in range(10):
        print("t", t)
        Loder[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
        u1 = torch.tensor(data[t]['train']['x'].reshape(-1, 3, 32, 32))  # .item()
        # print("u1",u1.size())

        TOTAL_NUM = u1.size()[0]
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        # u1.size()[0]
        # u2=torch.tensor(data[t]['train']['y'].reshape(-1))
        # u3 = data[t]['valid']['x']
        # print("u3",u3.size(),s)
        # u4=data[t]['valid']['y']
        dataset_new_train = Data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = Data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=10,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
        )
        Loder[t]['train'] = train_loader
        Loder[t]['test'] = test_loader
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #  test_dataset = datasets.CIFAR100('./data/', train=False, download=True, transform=transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean,
    #                                                 std)]))  # Data.TensorDataset(data[10//t_num]['test']['x'], data[10//t_num]['test']['y'])

    test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(), transforms.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    print("Loder is prepared")
    return data, taskcla[:10 // data[0]['ncla']], size, Loder, test_loader