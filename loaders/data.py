from functools import partial
import os
import gc
import torch
import logging
import torchtext
import torchvision
import transformers
import concurrent.futures
from src.datasets.fedif import FedIFGoldenDataset
from src import TqdmToLogger, stratified_split
from src.datasets import *
from src.loaders.split import simulate_split
import torchvision.transforms as transforms
from transformers import BertTokenizer
from torch.utils.data import Subset
import json
from src.robust_noise import gen_noisy_clients_vector, LabelNoiseWrapper, DataNoiseWrapper,PairNoiseWrapper
logger = logging.getLogger(__name__)

MEANS = {
    'CIFAR100': [0.5071, 0.4865, 0.4409],
}

STDS = {
    'CIFAR100': [0.2673, 0.2564, 0.2762],
}

VOCABS = {
    'Flickr30k': 'vocab.txt',
    'MedicalAbstracts': 'vocab.txt'
}

    

class SubsetWrapper(torch.utils.data.Dataset):
    """Wrapper of `torch.utils.data.Subset` module for applying individual transform.
    """
    def __init__(self, subset, suffix):
        self.subset = subset
        self.suffix = suffix

    def __getitem__(self, index):
        batch = self.subset[index]
        return batch

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset.dataset)} {self.suffix}'

def load_dataset(args, server=False):
    """Fetch and split requested datasets.
    
    Args:
        args: arguments
        
    Returns:
        split_map: {client ID: [assigned sample indices]}
            ex) {0: [indices_1], 1: [indices_2], ... , K: [indices_K]}
        server_testset: (optional) holdout dataset located at the central server, 
        client datasets: [(local training set, local test set)]
            ex) [tuple(local_training_set[indices_1], local_test_set[indices_1]), tuple(local_training_set[indices_2], local_test_set[indices_2]), ...]

    """
    TOKENIZER_STRINGS = {
        'DistilBert': 'distilbert-base-uncased',
        'SqueezeBert': 'squeezebert/squeezebert-uncased',
        'MobileBert': 'google/mobilebert-uncased'
    } 
    
    # error manager
    def _check_and_raise_error(entered, targeted, msg, eq=True):
        if eq:
            if entered == targeted: # raise error if eq(==) condition meets
                err = f'[{args.dataset.upper()}] `{entered}` {msg} is not supported for this dataset!'
                logger.exception(err)
                raise AssertionError(err)
        else:
            if entered != targeted: # raise error if neq(!=) condition meets
                err = f'[{args.dataset.upper()}] `{targeted}` {msg} is only supported for this dataset!'
                logger.exception(err)
                raise AssertionError(err)

    # method to get transformation chain
    def _get_transform(args, train=False, target=False, n_channels=3, to_pil_first=False, dataset=None):

        # NOTE: target tranform may be different from input transform, disable for both now
        if n_channels == 3:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage() if to_pil_first else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.Resize((args.resize, args.resize)) if args.resize is not None\
                        else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.RandomCrop(args.crop, pad_if_needed=True, padding=4) if (args.crop is not None and train)\
                        else torchvision.transforms.CenterCrop(args.crop) if (args.crop is not None and not train)\
                            else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.RandomRotation(args.randrot) if (args.randrot is not None and train)\
                        else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.RandomHorizontalFlip(args.randhf) if (args.randhf is not None and train)\
                        else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.RandomVerticalFlip(args.randvf) if (args.randvf is not None and train)\
                        else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.ColorJitter(brightness=args.randjit, contrast=args.randjit) if (args.randjit is not None and train)\
                        else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if args.imnorm and dataset is None and not target\
                        else torchvision.transforms.Normalize(mean=MEANS[dataset], std=STDS[dataset]) if args.imnorm and dataset is not None and not target\
                            else torchvision.transforms.Lambda(lambda x: x)
                ]
            )
        elif n_channels == 1:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage() if to_pil_first else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.Resize((args.resize, args.resize)) if args.resize is not None\
                        else torchvision.transforms.Lambda(lambda x: x),
                    # torchvision.transforms.RandomCrop(args.crop, pad_if_needed=True) if (args.crop is not None and train)\
                    #     else torchvision.transforms.CenterCrop(args.crop) if (args.crop is not None and not train)\
                    #         else torchvision.transforms.Lambda(lambda x: x),
                    # torchvision.transforms.RandomRotation(args.randrot) if (args.randrot is not None and train)\
                    #     else torchvision.transforms.Lambda(lambda x: x),
                    # torchvision.transforms.RandomHorizontalFlip(args.randhf) if (args.randhf is not None and train)\
                    #     else torchvision.transforms.Lambda(lambda x: x),
                    # torchvision.transforms.RandomVerticalFlip(args.randvf) if (args.randvf is not None and train)\
                    #     else torchvision.transforms.Lambda(lambda x: x),
                    # torchvision.transforms.ColorJitter(brightness=args.randjit, contrast=args.randjit) if (args.randjit is not None and train)\
                    #     else torchvision.transforms.Lambda(lambda x: x),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]) if args.imnorm and not target\
                        else torchvision.transforms.Lambda(lambda x: x)
                ]
            )
        return transform
    
    # method to construct per-client dataset
    def _construct_dataset(raw_train, idx, sample_indices, gamma_c_i: float = 0.0):
        task = raw_train.task if hasattr(raw_train, 'task') else None
        modality = raw_train.modality if hasattr(raw_train, 'modality') else None
        name = raw_train.name if hasattr(raw_train, 'name') else None
        subset = torch.utils.data.Subset(raw_train, sample_indices)

        if args.test_size == -1:
            training_set = subset
        else:
            if args.num_classes is None: # regression
                training_set, test_set = torch.utils.data.random_split(subset, [len(subset) - int(len(subset) * args.test_size), int(len(subset) * args.test_size)])
            else: # classification
                training_set, test_set = stratified_split(subset, args.test_size)

        traininig_set = SubsetWrapper(training_set, f'< {str(idx).zfill(8)} > (train)')

        # ✅ 把 client 噪声信息挂在 dataset 上（后面 server/client 会用到）
        traininig_set.noise_gamma = float(gamma_c_i)
        traininig_set.is_noisy = bool(gamma_c_i > 0)
        traininig_set.noise_type = int(getattr(args, "noise", 0))

        # ✅ 根据 noise 类型注入（只动训练集，不动 test）
        if getattr(args, "noise", 0) == 1:
            # 1) cls: Label Noise
            if (task == "cls") and (getattr(args, "num_classes", None) is not None) and gamma_c_i > 0:
                traininig_set = LabelNoiseWrapper(
                    traininig_set,
                    num_classes=int(args.num_classes),
                    gamma=float(gamma_c_i),
                    seed=int(args.seed) + int(idx) * 13,
                )
                traininig_set.noise_gamma = float(gamma_c_i)
                traininig_set.is_noisy = True
                traininig_set.noise_type = 1

            # 2) rtv(img+txt): Pair Noise（COCO/Flickr）
            elif (modality == "img+txt") and gamma_c_i > 0:
                traininig_set = PairNoiseWrapper(
                    traininig_set,
                    gamma=float(gamma_c_i),
                    seed=int(args.seed) + int(idx) * 13,
                    mode="swap_txt",  # 先固定
                )
                traininig_set.noise_gamma = float(gamma_c_i)
                traininig_set.is_noisy = True
                traininig_set.noise_type = 1

        elif getattr(args, "noise", 0) == 2:
            # DN：img/txt/img+txt 都可以用
            if gamma_c_i > 0:
                traininig_set = DataNoiseWrapper(
                    traininig_set,
                    modality=modality,
                    gamma=float(gamma_c_i),
                    mean=float(getattr(args, "level_n_mean", 0.0)),
                    std=float(getattr(args, "level_n_std", 0.1)),
                    seed=int(args.seed) + int(idx) * 17,
                    clip_min=float(getattr(args, "noise_clip_min", -1.0)),
                    clip_max=float(getattr(args, "noise_clip_max", 1.0)),
                    txt_drop_prob=float(getattr(args, "txt_drop_prob", 0.1)),
                    txt_pad_id=int(getattr(args, "txt_pad_id", 0)),
                )
                traininig_set.noise_gamma = float(gamma_c_i)
                traininig_set.is_noisy = True
                traininig_set.noise_type = 2
        if len(subset) * args.test_size > 0:
            test_set = SubsetWrapper(test_set, f'< {str(idx).zfill(8)} > (test)')
        else:
            test_set = None
        return (traininig_set, test_set, task, modality, name)
    
    #################
    # base settings #
    #################
    # required intermediate outputs
    raw_train, raw_test = None, None

    # required outputs
    split_map, client_datasets = None, None
    
    # optional argument for data transforms
    transforms = [None, None]
    
    ####################
    # for text dataset #
    ####################
    tokenizer = None
    if args.use_model_tokenizer or args.use_pt_model:
        assert args.model_name in ['DistilBert', 'SqueezeBert', 'MobileBert'], 'Please specify a proper model!'

    if args.use_model_tokenizer:
        assert args.model_name.lower() in transformers.models.__dict__.keys(), f'Please check if the model (`{args.model_name}`) is supported by `transformers` module!'
        module = transformers.models.__dict__[f'{args.model_name.lower()}']
        tokenizer = getattr(module, f'{args.model_name}Tokenizer').from_pretrained(TOKENIZER_STRINGS[args.model_name])

    if args.use_bert_tokenizer:
        if args.dataset in VOCABS.keys():
            tokenizer = BertTokenizer(os.path.join(args.data_path, VOCABS[args.dataset]))
        else:
            tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case="uncased" in 'bert_base_uncased'
        )
    #################
    # fetch dataset #
    #################
    logger.info(f'[LOAD] Fetch dataset!')
    
    if args.dataset in ['FEMNIST', 'Shakespeare', 'Sent140', 'CelebA', 'Reddit']: # 1) for a special dataset - LEAF benchmark...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
         
        # define transform
        if args.dataset in ['FEMNIST', 'CelebA']:
            # check if `crop` is required
            if args.crop is None:
                logger.info(f'[LOAD] Dataset `{args.dataset}` may require `crop` argument; (recommended: `FEMNIST` - 28, `CelebA` - 84)!')
            transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        elif args.dataset == 'Reddit':
            args.rawsmpl = 1.0

        # construct split hashmap, client datasets
        # NOTE: for LEAF benchmark, values of `split_map` hashmap is not indices, but sample counts of tuple (training set, test set)!
        split_map, client_datasets, args = fetch_leaf(
            args=args,
            dataset_name=args.dataset, 
            root=args.data_path, 
            seed=args.seed, 
            raw_data_fraction=args.rawsmpl, 
            test_size=args.test_size, 
            transforms=transforms
        )

        # no global holdout set for LEAF
        raw_test = None  
    elif args.dataset == 'Flickr30k':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_flickr30k(args=args, root=args.data_path, transforms=transforms, tokenizer=tokenizer)
    
    elif args.dataset == 'Coco':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_coco(args=args, root=args.data_path, transforms=transforms, tokenizer=tokenizer)
    

    elif args.dataset in torchvision.datasets.__dict__.keys(): # 3) for downloadable datasets in `torchvision.datasets`...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True, dataset=args.dataset), _get_transform(args, train=False, dataset=args.dataset)]
        raw_train, raw_test, args = fetch_torchvision_dataset(args=args, dataset_name=args.dataset, root=args.data_path, transforms=transforms)
        
    elif args.dataset in torchtext.datasets.__dict__.keys(): # 4) for downloadable datasets in `torchtext.datasets`...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, args = fetch_torchtext_dataset(args=args, dataset_name=args.dataset, root=args.data_path, seq_len=args.seq_len, tokenizer=tokenizer, num_embeddings=args.num_embeddings) 
        
    elif args.dataset == 'TinyImageNet': # 5) for other public datasets...
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_tinyimagenet(args=args, root=args.data_path, transforms=transforms)
        
    elif args.dataset == 'CINIC10':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True), _get_transform(args, train=False)]
        raw_train, raw_test, args = fetch_cinic10(args=args, root=args.data_path, transforms=transforms)
    
    elif args.dataset == 'SpeechCommands':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, args = fetch_speechcommands(args=args, root=args.data_path)

    elif 'BeerReviews' in args.dataset:
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        aspect_type = {'A': 'aroma', 'L': 'look'}
        parsed_type = args.dataset[-1]
        if parsed_type in ['A', 'L']:
            aspect = aspect_type[parsed_type]
        else:
            err = '[LOAD] Please check dataset name!'
            logger.exception(err)
            raise Exception(err)
        raw_train, raw_test, args = fetch_beerreviews(args=args, root=args.data_path, aspect=aspect, tokenizer=tokenizer)  
        
    elif args.dataset == 'Heart':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_heart(args=args, root=args.data_path, seed=args.seed, test_size=args.test_size)
    
    elif args.dataset == 'Adult':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_adult(args=args, root=args.data_path, seed=args.seed, test_size=args.test_size)
    
    elif args.dataset == 'Cover':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_cover(args=args, root=args.data_path, seed=args.seed, test_size=args.test_size)  
    
    elif args.dataset == 'GLEAM':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        split_map, client_datasets, args = fetch_gleam(args=args, root=args.data_path, seed=args.seed, test_size=args.test_size, seq_len=args.seq_len)

    elif args.dataset == 'BraTS':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True, n_channels=1, to_pil_first=True), _get_transform(args, train=False, n_channels=1, to_pil_first=True), _get_transform(args,target=True, n_channels=1, to_pil_first=True)]
        raw_train, raw_test, args = fetch_brats(args=args, root=args.data_path, transforms=transforms, modality=args.modality)
    
    elif args.dataset == 'MedMNIST':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [_get_transform(args, train=True, n_channels=1), _get_transform(args, train=False, n_channels=1), _get_transform(args,target=True, n_channels=1)]
        raw_train, raw_test, args = fetch_medmnist(args=args, root=args.data_path, transforms=transforms, modality=args.modality)
    
    elif args.dataset == 'MTSamples':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [partial(tokenizer, padding='max_length', max_length=args.seq_len, truncation=True), partial(tokenizer, padding='max_length', max_length=args.seq_len, truncation=True)]
        raw_train, raw_test, args = fetch_mtsamples(args=args, root=args.data_path, transforms=transforms, modality=args.modality)
    elif args.dataset == 'MedicalAbstracts':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        transforms = [partial(tokenizer, padding='max_length', max_length=args.seq_len, truncation=True), partial(tokenizer, padding='max_length', max_length=args.seq_len, truncation=True)]
        raw_train, raw_test, args = fetch_medabstracts(args=args, root=args.data_path, transforms=transforms, modality=args.modality)
    
    else: # x) for a dataset with no support yet or incorrectly entered...
        err = f'[LOAD] Dataset `{args.dataset}` is not supported or seems incorrectly entered... please check!'
        logger.exception(err)
        raise Exception(err)     
    logger.info(f'[LOAD] ...successfully fetched dataset!')

    if server:
        return (raw_train, raw_test)
    
    ############
    # finalize #
    ############
    # check if global holdout set is required or not
    # if args.eval_type == 'local':
    #     if args.test_size == -1: 
    #         assert raw_test is not None
    #         _raw_test = raw_test
    #     raw_test = None
    # else:
    #     if raw_test is None:
    #         err = f'[LOAD] Dataset `{args.dataset.upper()}` does not support pre-defined validation/test set, which can be used for `global` evluation... please check! (current `eval_type`=`{args.eval_type}`)'
    #         logger.exception(err)
    #         raise AssertionError(err)
            
    # get split indices if None
    if split_map is None:
        logger.info(f'[SIMULATE] Simulate dataset split (split scenario: `{args.split_type.upper()}`)!')
        split_map = simulate_split(args, raw_train)
        logger.info(f'[SIMULATE] ...done simulating dataset split (split scenario: `{args.split_type.upper()}`)!')
    
    # construct client datasets if None
    if client_datasets is None:
        logger.info(f'[SIMULATE] Create client datasets!')
        client_datasets = []
        validation_sets = []
        gamma_c, gamma_s = gen_noisy_clients_vector(args, num_users=args.K)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.K, os.cpu_count() - 1)) as workhorse:
        for idx, sample_indices in TqdmToLogger(
            enumerate(split_map.values()), 
            logger=logger, 
            desc=f'[SIMULATE] ...creating client datasets... ',
            total=len(split_map)
            ):
            res = _construct_dataset(raw_train, idx, sample_indices, gamma_c[idx])
            validation_sets.append(res[1])
            client_datasets.append(res) 
        logger.info(f'[SIMULATE] ...successfully created client datasets!')
        
        ## //when if assigning pre-defined test split as a local holdout set (just divided by the total number of clients)
        if (args.eval_type == 'local'):  
            holdout_sets = torch.utils.data.random_split(raw_test, [int(len(raw_test) / args.K)  for _ in range(args.K)])
            holdout_sets = [SubsetWrapper(holdout_set, f'< {str(idx).zfill(8)} > (test)') for idx, holdout_set in enumerate(holdout_sets)]
            augmented_datasets = []
            for idx, client_dataset in enumerate(client_datasets): 
                augmented_datasets.append((client_dataset[0], holdout_sets[idx], client_dataset[2], client_dataset[3]))
            client_datasets = augmented_datasets
    gc.collect()
    return raw_test, client_datasets, validation_sets

def load_index_golden(json_path, base_dataset):
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Subset(base_dataset, obj["indices"])

def load_datasets(args):
    """Fetch and split requested datasets. Enhanced for FedIF."""

    # [Fix 1] 保存原始数据根目录，防止在循环中 args.data_path 被修改导致后续路径错误
    root_data_path = args.data_path

    datasets = args.datasets
    modalities = args.modalities
    data_paths = args.data_paths
    num_clients = args.Ks
    num_datasets = len(datasets) - 1  # Server is the last one

    if len(num_clients) == 1:
        num_clients = [num_clients[0]] * num_datasets

    raw_test = None
    client_datasetss = []

    # raw_tests 是一个字典，用于存储所有任务的验证集以及 FedIF 的黄金数据集
    # 结构: {'CIFAR100': Dataset, 'AG_NEWS': Dataset, 'fedif_golden_v0': Dataset, ...}
    raw_tests = {}

    # --- 1. 加载各个任务的数据集 ---
    for i in range(num_datasets):
        args.dataset = datasets[i]
        args.data_path = data_paths[i]  # 这里修改了 args.data_path
        args.modality = modalities[i]
        args.K = int(num_clients[i])

        # load_dataset 返回: raw_test, client_datasets, validation_sets
        server_dataset, client_datasets, validation_sets = load_dataset(args)

        # 将该任务的验证集放入字典
        raw_tests[datasets[i]] = server_dataset

        if i == 0:
            client_datasetss = client_datasets
        else:
            for client_dataset in client_datasets:
                client_datasetss.append(client_dataset)

    # --- 2. [新增] 加载多版本 FedIF 黄金数据集 ---
    print(f"[Data] Scanning for FedIF Golden Sets in: {root_data_path}...")

    # 准备 Tokenizer 和 Transform (只需初始化一次)
    try:
        # 尝试加载 BERT Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 准备图像变换 (标准 ImageNet 归一化)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    except Exception as e:
        print(f"[Data] Warning: Failed to init tokenizer/transform for FedIF: {e}")
        tokenizer = None
        val_transform = None

    if tokenizer and val_transform:
        found_golden = False

        # 搜索路径优先级：Coco目录 > Flickr目录 > 根目录
        search_dirs = [
            os.path.join(root_data_path, 'coco'),
            os.path.join(root_data_path, 'flickr30k'),
            root_data_path
        ]

        # A. 优先尝试查找多版本文件 (fedif_golden_set_v0.json, _v1.json ...)
        for base_dir in search_dirs:
            if not os.path.exists(base_dir): continue

            versions_found = []
            # 扫描 v0 到 v9
            for v in range(10):
                p = os.path.join(base_dir, f'fedif_golden_set_v{v}.json')
                if os.path.exists(p) and os.path.isfile(p):
                    versions_found.append((v, p))

            if versions_found:
                print(f"[Data] ★ Found {len(versions_found)} Golden Set versions in: {base_dir}")
                for v, p in versions_found:
                    key_name = f'fedif_golden_v{v}'  # e.g., fedif_golden_v0
                    try:
                        d_set = FedIFGoldenDataset(
                            json_file=p,
                            transform=val_transform,
                            tokenizer=tokenizer,
                            max_length=getattr(args, 'seq_len', 40)
                        )
                        raw_tests[key_name] = d_set
                    except Exception as e:
                        print(f"[Data] Failed to load {p}: {e}")

                found_golden = True
                break  # 找到了版本文件就不再搜其他目录

        # B. 回退机制：如果没有找到版本文件，尝试找单文件 (fedif_golden_set.json)
        if not found_golden:
            target_golden_path = None
            for p in [os.path.join(d, 'fedif_golden_set.json') for d in search_dirs]:
                if os.path.exists(p) and os.path.isfile(p):
                    target_golden_path = p
                    break

            if target_golden_path:
                print(f"[Data] ★ Found single Golden Set: {target_golden_path}")
                try:
                    d_set = FedIFGoldenDataset(
                        json_file=target_golden_path,
                        transform=val_transform,
                        tokenizer=tokenizer,
                        max_length=getattr(args, 'seq_len', 40)
                    )
                    raw_tests['fedif_golden'] = d_set  # 使用旧 Key
                except Exception as e:
                    print(f"[Data] Failed to load {target_golden_path}: {e}")
            else:
                print("[Data] ⚠️ No FedIF Golden Set found. FedIF influence calculation may be skipped.")

    # --- 3. 加载 Server 端数据 (最后一个 dataset) ---
    gold_dir = os.path.join(root_data_path, "golden_cls")
    if os.path.isdir(gold_dir):
        print(f"[Data] Scanning for CLS Golden Sets in: {gold_dir}...")

        # 注意：这里的 key 必须和 raw_tests 里 dataset 的 key 完全一致
        # 你 raw_tests 的 key 是 datasets[i]（例如 "CIFAR100" / "AG_NEWS"）
        if "CIFAR100" in raw_tests:
            for v in range(10):
                p = os.path.join(gold_dir, f"CIFAR100_golden_v{v}.json")
                if os.path.exists(p):
                    raw_tests[f"CIFAR100_golden_v{v}"] = load_index_golden(p, raw_tests["CIFAR100"])
            print("[Data] Loaded CIFAR100 golden:", [k for k in raw_tests if "CIFAR100_golden" in k])

        if "AG_NEWS" in raw_tests:
            for v in range(10):
                p = os.path.join(gold_dir, f"AG_NEWS_golden_v{v}.json")
                if os.path.exists(p):
                    raw_tests[f"AG_NEWS_golden_v{v}"] = load_index_golden(p, raw_tests["AG_NEWS"])
            print("[Data] Loaded AG_NEWS golden:", [k for k in raw_tests if "AG_NEWS_golden" in k])
    else:
        print(f"[Data] No golden_cls folder found at {gold_dir}, skip CLS golden loading.")
    args.dataset = datasets[-1]
    args.data_path = data_paths[-1]
    args.modality = modalities[-1]
    args.K = 1

    server_datasets = load_dataset(args, server=True)
    args.K = sum([int(num_client) for num_client in num_clients])

    # --- 4. [Fix 2] 调整返回顺序 ---
    # 返回 (raw_tests字典, server_datasets元组)
    # 这样 FedifServer 拿到 datasets[0] 就是字典，datasets[1] 就是元组
    return (raw_tests, server_datasets), client_datasetss
