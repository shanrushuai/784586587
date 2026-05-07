import os
import sys
import time
import torch
import argparse
import traceback
import wandb

from importlib import import_module
# from torch.utils.tensorboard import SummaryWriter

from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model, load_datasets

# Uncomment this if there is deadlock in DataLoader
# torch.set_num_threads(16)

def main(args, writer):
    """Main program to run federated learning.
    
    Args:
        args: user input arguments parsed by argparser
        writer: `torch.utils.tensorboard.SummaryWriter` instance for TensorBoard tracking
    """
    # set seed for reproducibility
    set_seed(args.seed)

    # get dataset
    server_dataset, client_datasets = load_dataset(args) if not args.multi_task else load_datasets(args)

    # check all args before FL
    args = check_args(args)
    
    # get model # NOTE: removed since modes are not identical in each clients
    # model, args = load_model(args)

    # create central server
    server_class = import_module(f'src.server.{args.algorithm}server').__dict__[f'{args.algorithm.title()}Server']
    server = server_class(args=args, writer=writer, server_dataset=server_dataset, client_datasets=client_datasets, model_str=args.model_name)
    
    # federated learning
    for curr_round in range(1, args.R + 1):
        ## update round indicator
        server.round = curr_round

        ## update after sampling clients randomly
        selected_ids = server.update() 

        ## evaluate on clients not sampled (for measuring generalization performance)
        if (curr_round % args.eval_every == 0) or (curr_round == args.R):
            server.evaluate([])
    else:
        ## wrap-up
        server.finalize()



if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    
    #####################
    # Default arguments #
    #####################
    parser.add_argument('--goal', help='goal of the experiment', type=str, default=None)
    parser.add_argument('--exp_name', help='name of the experiment', type=str, required=True)
    parser.add_argument('--seed', help='global random seed', type=int, default=5959)
    parser.add_argument('--server_device', help='device to use; `cpu`, `cuda`, `cuda:GPU_NUMBER`', type=str, default=f'cuda:{torch.cuda.device_count() - 1}')
    parser.add_argument('--data_path', help='path to save & read raw data', type=str, default='./data')
    parser.add_argument('--modality', help='modality of the dataset', type=str, default='ct')
    parser.add_argument('--log_path', help='path to save logs', type=str, default='./log')
    parser.add_argument('--result_path', help='path to save results', type=str, default='./result')
    parser.add_argument('--use_tb', help='use TensorBoard for log tracking (if passed)', action='store_true')
    parser.add_argument('--tb_port', help='TensorBoard port number (valid only if `use_tb`)', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address (valid only if `use_tb`)', type=str, default='0.0.0.0')
    parser.add_argument('--distributed', help='enable distributed training', action='store_true')
    parser.add_argument('--mm_distributed', help='enable distributed training for mm clients', action='store_true')
    parser.add_argument('--mp', help='enable multi-processing instead of multi-threading', action='store_true')


    
    #####################
    # Dataset arguments #
    #####################
    ## dataset configuration arguments (For MFL, none of these are used, please use --datasets, etc.)
    parser.add_argument('--dataset', help='''name of dataset to use for an experiment (NOTE: case sensitive)
    - image classification datasets in `torchvision.datasets`,
    - text classification datasets in `torchtext.datasets`,
    - LEAF benchmarks [ FEMNIST | Sent140 | Shakespeare | CelebA | Reddit ],
    - among [ TinyImageNet | CINIC10 | SpeechCommands | BeerReviewsA | BeerReviewsL | Heart | Adult | Cover | GLEAM ]
    ''', type=str)
    parser.add_argument('--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)', type=float, choices=[Range(-1, 1.)], default=0.2)
    parser.add_argument('--rawsmpl', help='a fraction of raw data to be used (valid only if one of `LEAF` datasets is used)', type=float, choices=[Range(0., 1.)], default=1.0)
    
    ## data augmentation arguments
    parser.add_argument('--resize', help='resize input images (using `torchvision.transforms.Resize`)', type=int, default=None)
    parser.add_argument('--crop', help='crop input images (using `torchvision.transforms.CenterCrop` (for evaluation) and `torchvision.transforms.RandomCrop` (for training))', type=int, default=None)
    parser.add_argument('--imnorm', help='normalize channels with mean 0.5 and standard deviation 0.5 (using `torchvision.transforms.Normalize`, if passed)', action='store_true')
    parser.add_argument('--randrot', help='randomly rotate input (using `torchvision.transforms.RandomRotation`)', type=int, default=None)
    parser.add_argument('--randhf', help='randomly flip input horizontaly (using `torchvision.transforms.RandomHorizontalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randvf', help='randomly flip input vertically (using `torchvision.transforms.RandomVerticalFlip`)', type=float, choices=[Range(0., 1.)], default=None)
    parser.add_argument('--randjit', help='randomly change the brightness and contrast (using `torchvision.transforms.ColorJitter`)', type=float, choices=[Range(0., 1.)], default=None)

    ## statistical heterogeneity simulation arguments
    parser.add_argument('--split_type', help='''type of data split scenario
    - `iid`: statistically homogeneous setting,
    - `unbalanced`: unbalanced in sample counts across clients,
    - `patho`: pathological non-IID split scenario proposed in (McMahan et al., 2016),
    - `diri`: Dirichlet distribution-based split scenario proposed in (Hsu et al., 2019),
    - `pre`: pre-defined data split scenario
    ''', type=str, choices=['iid', 'unbalanced', 'patho', 'diri', 'pre'], required=True)
    parser.add_argument('--mincls', help='the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)', type=int, default=2)
    parser.add_argument('--cncntrtn', help='a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)', type=float, default=0.1)
    
    ########################
    # Multi-task arguments #
    ########################
    parser.add_argument('--multi-task', help='If it is multi-task learning. For MFL, it should always be True.', action='store_true') # 
    parser.add_argument('--debug', help='reduce training for debugging', action='store_true')
    parser.add_argument('--pretrained', help='use pre-trained model', action='store_true')
    parser.add_argument('--datasets', help='''name of datasets to use for an experiment (NOTE: case sensitive)
    for multi-task learning, last one is the server datasaet.
    ''', nargs='+', type=str)
    parser.add_argument('--data_paths', help='''name of datasets to use for an experiment (NOTE: case sensitive)
     for multi-task learning.
    ''', nargs='+', type=str)
    parser.add_argument('--modalities', help='''name of modalities to use for an experiment (NOTE: case sensitive)
     for multi-task learning.
    ''', nargs='+', type=str)
    parser.add_argument('--Ks', help='number of total cilents participating in federated training for each task)', type=int, default=1, nargs='+', ) # Can be skipped now

    parser.add_argument('--Cs', help='number of total cilents participating in federated training for each task)', type=float, default=0.25, nargs='+', ) # Can be skipped now

    parser.add_argument('--shared_param', help='''strategy for multi-task learning.
    ''', type=str)
    parser.add_argument('--share_scope', help='''strategy for multi-task learning.
    ''', type=str)
    parser.add_argument('--colearn_param', help='''strategy for multi-task learning.
    ''', type=str)

    parser.add_argument('--compensation', help='modality compensation', action='store_true')

    parser.add_argument('--reduce_samples', help='reduce samples for each task.', type=int, default=50000)
    parser.add_argument('--reduce_test_samples', help='reduce samples for each task.', type=int, default=-1)

    parser.add_argument('--reduce_samples_seg_scale', help='reduce samples for segmentation.', type=float, default=-1)
    parser.add_argument('--reduce_samples_cls_scale', help='reduce samples for classification.', type=float, default=-1)


    parser.add_argument('--num_thread', help='number of thread', type=int, default=1)

    parser.add_argument('--num_transformer_layers', help='number of layers of the transformer', type=int, default=12)

    ########################
    # Multi-modal arguments #
    ########################

    parser.add_argument('--equal_sampled', help='sample same number of client per round', action='store_true')

    ########################
    # FedCola arguments     #
    ########################

    parser.add_argument('--warmup_modality', type=str, default='none',
                        help='warm up modality')
    parser.add_argument('--warmup_rounds', type=int, default=5,
                        help='warm up rounds')
    parser.add_argument('--freeze_modality', type=str, default='none',
                        help='warm up modality')
    parser.add_argument('--freeze_rounds', type=int, default=5,
                        help='warm up rounds')
    parser.add_argument('--out_modality_scales', help='constant to control the in-modal aggregation)', type=str, default='[1]',)
    parser.add_argument('--fedavg_eval', help='use fedavg for evaluation', action='store_true')

    parser.add_argument('--with_aux', help='use aux modality', action='store_true')
    parser.add_argument('--aux_trained', help='if aux is trained', action='store_true')
    parser.add_argument('--aux_attn_only', help='if aux is trained', action='store_true')
    parser.add_argument('--aux_mlp_only', help='if aux is trained', action='store_true')



    parser.add_argument('--flickr_train_all', help='use all flickr data. If false, use 10000', action='store_true')

    ########################
    # Vector arguments     #
    ########################

    parser.add_argument('--v_epoch', help='epochs to trian the lambda', type=int, default=5)
    parser.add_argument('--supervised', help='use supervised vector', action='store_true')
    parser.add_argument('--train_as_val', help='use supervised vector', action='store_true')

    ########################
    # CreamFL arguments     #
    ########################
    parser.add_argument('--pub_data_dir', type=str, default='data/coco/all_images/', help='public data directory')
    parser.add_argument('--pub_anno_path', type=str, default='data/coco/annotations/captions_val2014.json',
                        help='public annotation directory')
    parser.add_argument('--num_pub_samples', type=int, default=500,
                        help='public samples number')
    parser.add_argument('--pub_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--p_lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--interintra_weight', type=float, default=0.5,
                        help='inter intra loss weight')
    parser.add_argument('--kd_weight', type=float, default=0.3, help='coefficient of kd')
    parser.add_argument('--no_mm_contrastive', action='store_true', help='no local contrastive for mm clients')

    ########################
    # FedIoT arguments    #
    ########################
    parser.add_argument('--mm_scale', type=float, default=100, help='coefficent for multi-modal clients')
    parser.add_argument('--no_intra', action='store_true', help='Disable Intra-Weighting')
    parser.add_argument('--no_sscf', action='store_true', help='Disable SSCF')
    parser.add_argument('--soft_update_tau', type=float, default=0.5,
                        help='Soft update factor for FedIF global aggregation (default: 0.5)')
    parser.add_argument('--align_temp', type=float, default=0.2,
                        help='Temperature for contrastive loss in FedIF (default: 0.2)')

    ###################
    # Model arguments #
    ###################
    ## model
    parser.add_argument('--model_name', help='a model to be used (NOTE: case sensitive)', type=str,
        required=True
    )
    parser.add_argument('--hidden_size', help='hidden channel size for vision models, or hidden dimension of language models', type=int, default=64)
    parser.add_argument('--dropout', help='dropout rate', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--use_model_tokenizer', help='use a model-specific tokenizer (if passed)', action='store_true')
    parser.add_argument('--use_bert_tokenizer', help='use a bert tokenizer', action='store_true')
    parser.add_argument('--vocab_size', help='size of the vocabulary', type=int, default=30522)
    parser.add_argument('--use_pt_model', help='use a pre-trained model weights for fine-tuning (if passed)', action='store_true')
    parser.add_argument('--seq_len', help='maximum sequence length used for `torchtext.datasets`)', type=int, default=40)
    parser.add_argument('--num_layers', help='number of layers in recurrent cells', type=int, default=2)
    parser.add_argument('--num_embeddings', help='size of an embedding layer', type=int, default=1000)
    parser.add_argument('--embedding_size', help='output dimension of an embedding layer', type=int, default=512)
    parser.add_argument('--init_type', help='weight initialization method', type=str, default='kaiming', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'truncnorm', 'none'])
    parser.add_argument('--init_gain', type=float, default=1.0, help='magnitude of variance used for weight initialization')
    
    ######################
    # Learning arguments #
    ######################
    ## federated learning settings
    parser.add_argument('--algorithm', help='federated learning algorithm to be used', type=str,
         choices=['fedavg', 'fedsgd', 'fedprox', 'fedavgm', 'vector', 'creamfl', 'fedif','fediot'],
        required=True
    )
    parser.add_argument('--eval_type', help='''the evaluation type of a model trained from FL algorithm
    - `local`: evaluation of personalization model on local hold-out dataset  (i.e., evaluate personalized models using each client\'s local evaluation set)
    - `global`: evaluation of a global model on global hold-out dataset (i.e., evaluate the global model using separate holdout dataset located at the server)
    - 'both': combination of `local` and `global` setting
    ''', type=str,
        choices=['local', 'global', 'both'],
        required=True
    )
    parser.add_argument('--eval_fraction', help='fraction of randomly selected (unparticipated) clients for the evaluation (valid only if `eval_type` is `local` or `both`)', type=float, choices=[Range(1e-8, 1.)], default=1.)
    parser.add_argument('--eval_every', help='frequency of the evaluation (i.e., evaluate peformance of a model every `eval_every` round)', type=int, default=1)
    parser.add_argument('--eval_metrics', help='metric(s) used for evaluation', type=str,
        choices=[
            'acc1', 'acc5', 'auroc', 'auprc', 'youdenj', 'f1', 'precision', 'recall',
            'seqacc', 'mse', 'mae', 'mape', 'rmse', 'r2', 'd2'
        ], nargs='+', required=True
    )
    parser.add_argument('--K', help='number of total cilents participating in federated training', type=int, default=100)
    parser.add_argument('--R', help='number of total federated learning rounds', type=int, default=1000)
    parser.add_argument('--C', help='sampling fraction of clients per round (full participation when 0 is passed)', type=float, choices=[Range(0., 1.)], default=0.1)
    parser.add_argument('--E', help='number of local epochs', type=int, default=5)
    parser.add_argument('--B', help='local batch size (full-batch training when zero is passed)', type=int, default=10)
    parser.add_argument('--eval_batch_size', help='eval batch size (full-batch training when zero is passed)', type=int, default=64)
    parser.add_argument('--beta1', help='server momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    # [Add this to main.py]
    # FedIF arguments

    # ---------------- Robustness (Noise) ----------------

    # optimization arguments
    parser.add_argument('--no_shuffle', help='do not shuffle data when training (if passed)', action='store_true')
    parser.add_argument('--optimizer', help='type of optimization method (NOTE: should be a sub-module of `torch.optim`, thus case-sensitive)', type=str, default='SGD', required=True)
    parser.add_argument('--max_grad_norm', help='a constant required for gradient clipping', type=float, choices=[Range(0., float('inf'))], default=0.)
    parser.add_argument('--weight_decay', help='weight decay (L2 penalty)', type=float, choices=[Range(0., 1.)], default=0)
    parser.add_argument('--momentum', help='momentum factor', type=float, choices=[Range(0., 1.)], default=0.)
    parser.add_argument('--nesterov', help='use Nesterov momentum (if passed)', action='store_true')
    parser.add_argument('--lr', help='learning rate for local updates in each client', type=float, choices=[Range(0., 100.)], default=0.01, required=True)
    parser.add_argument('--lr_decay', help='decay rate of learning rate', type=float, choices=[Range(0., 1.)], default=1.0)
    parser.add_argument('--lr_decay_step', help='intervals of learning rate decay', type=int, default=20)
    parser.add_argument('--criterion', help='objective function (NOTE: should be a submodule of `torch.nn`, thus case-sensitive)', type=str, required=True)
    parser.add_argument('--mu', help='constant for proximity regularization term (valid only if the algorithm is `fedprox`)', type=float, choices=[Range(0., 1e6)], default=0.01)
    # noise benchmark
    parser.add_argument('--noise', type=int, default=0)  # 0 clean, 1 LN, 2 DN, 3 GN, 4 ADV
    parser.add_argument('--level_n_system', type=float, default=0.4)
    parser.add_argument('--level_n_lowerb', type=float, default=0.5)
    parser.add_argument('--level_n_upperb', type=float, default=0.7)
    parser.add_argument('--level_n_mean', type=float, default=0.0)
    parser.add_argument('--level_n_std', type=float, default=0.1)

    parser.add_argument('--fedif_gamma', type=float, default=0.9,
                        help='EMA decay factor for FedIF influence scores (default: 0.9)')
    parser.add_argument('--enable_fisher', action='store_true',
                        help='Enable client-side diagonal Fisher estimation for shared attention.')
    parser.add_argument('--fisher_batches', type=int, default=2,
                        help='How many local batches to estimate Fisher (small number is enough).')
    parser.add_argument('--fisher_clip', type=float, default=10.0,
                        help='Clip Fisher values to avoid numerical explosion.')
    parser.add_argument('--fisher_norm_eps', type=float, default=1e-6,
                        help='Eps for Fisher normalization.')

    # =================== Resource-aware Selector (NEW) ===================
    parser.add_argument('--resource_selector', type=str, default='none',
                        choices=['none', 'oort', 'lyapunov', 'fedbalancer'],
                        help='Switchable resource-aware client selector.')
    parser.add_argument('--fedscale_capacity_path', type=str,
                        default='FedScale-master/benchmark/dataset/data/device_info/client_device_capacity',
                        help='Path to FedScale client_device_capacity pickle (falls back to synthetic).')
    parser.add_argument('--energy_per_flop', type=float, default=1.0e-10,
                        help='Joules per FLOP (alpha).')
    parser.add_argument('--energy_per_byte', type=float, default=1.0e-7,
                        help='Joules per uploaded byte (beta).')
    # Lyapunov-specific
    parser.add_argument('--lyap_V', type=float, default=3.0,
                        help='Lyapunov trade-off parameter V (quality vs resource queues).')
    parser.add_argument('--lyap_mu', type=float, default=0.1,
                        help='Momentum coefficient for virtual queue [0,0.3].')
    parser.add_argument('--lyap_beta', type=float, default=0.3,
                        help='UCB exploration strength in Bandit-Lyapunov.')
    parser.add_argument('--lyap_fisher_density', action='store_true',
                        help='Multiply quality by Fisher information density Psi_k.')
    parser.add_argument('--lyap_energy_weight', type=float, default=1.0,
                        help='Weight of normalized energy drift in RMAL-ED.')
    parser.add_argument('--lyap_delay_weight', type=float, default=1.0,
                        help='Weight of marginal round-latency drift in RMAL-ED.')
    parser.add_argument('--lyap_tail_weight', type=float, default=0.5,
                        help='Weight of CVaR tail-latency drift in RMAL-ED.')
    parser.add_argument('--lyap_energy_budget_scale', type=float, default=0.90,
                        help='Normalized energy budget scale; <1 tightens energy.')
    parser.add_argument('--lyap_time_budget_scale', type=float, default=1.00,
                        help='Normalized max-latency budget scale; <1 tightens delay.')
    parser.add_argument('--lyap_tail_budget_scale', type=float, default=1.05,
                        help='Tail-latency budget scale relative to modality deadlines.')
    parser.add_argument('--lyap_cvar_alpha', type=float, default=0.80,
                        help='CVaR confidence level for tail-latency queue.')
    parser.add_argument('--lyap_resource_clip', type=float, default=4.0,
                        help='Clip robust-normalized energy/time to [1/x, x].')
    parser.add_argument('--lyap_resource_floor', type=float, default=0.20,
                        help='Minimum resource shadow price before queues build up.')
    parser.add_argument('--lyap_mean_norm', action='store_true',
                        help='Use mean normalization instead of robust median-MAD.')
    parser.add_argument('--lyap_static_V', action='store_true',
                        help='Disable adaptive V_t and use fixed --lyap_V.')
    parser.add_argument('--lyap_V_min', type=float, default=1.0,
                        help='Lower bound for adaptive V_t.')
    parser.add_argument('--lyap_V_max', type=float, default=6.0,
                        help='Upper bound for adaptive V_t.')
    parser.add_argument('--lyap_adapt_energy_coeff', type=float, default=0.5,
                        help='Adaptive V_t sensitivity to energy queue pressure.')
    parser.add_argument('--lyap_adapt_time_coeff', type=float, default=0.5,
                        help='Adaptive V_t sensitivity to latency queue pressure.')
    parser.add_argument('--lyap_adapt_tail_coeff', type=float, default=0.3,
                        help='Adaptive V_t sensitivity to tail-latency queue pressure.')
    # Oort-specific
    parser.add_argument('--oort_penalty', type=float, default=2.0,
                        help='Round duration penalty exponent for Oort.')
    parser.add_argument('--oort_pref_pct', type=float, default=80.0,
                        help='Percentile for preferred duration.')
    parser.add_argument('--oort_ucb', type=float, default=0.1,
                        help='UCB exploration coefficient for Oort.')
    # FedBalancer-specific
    parser.add_argument('--fb_alpha', type=float, default=2.0,
                        help='Deadline penalty exponent for FedBalancer.')
    parser.add_argument('--fb_epsilon', type=float, default=0.9,
                        help='Initial exploration ratio for FedBalancer.')
    parser.add_argument('--fb_ddl_step', type=float, default=0.05,
                        help='Deadline ratio step size for FedBalancer.')
    parser.add_argument('--fb_window', type=int, default=5,
                        help='Adaptive control window for FedBalancer.')
    # noise benchmark

    # DN for text
    parser.add_argument('--txt_drop_prob', type=float, default=0.1)
    parser.add_argument('--txt_pad_id', type=int, default=0)
    parser.add_argument('--noise_clip_min', type=float, default=-1.0)
    parser.add_argument('--noise_clip_max', type=float, default=1.0)

    # GN (recommended smaller than level_n_std)
    parser.add_argument('--grad_n_mean', type=float, default=0.0)
    parser.add_argument('--grad_n_std', type=float, default=1e-4)

    # ADV
    parser.add_argument('--adv_eps', type=float, default=0.03)
    parser.add_argument('--adv_clip_min', type=float, default=-1.0)
    parser.add_argument('--adv_clip_max', type=float, default=1.0)
    # parse arguments
    args = parser.parse_args()
    args.out_modality_scales = eval(args.out_modality_scales)
    if len(args.out_modality_scales) == 1:
        args.out_modality_scales = args.out_modality_scales * (len(args.modalities) - 1)
    
    # make path for saving losses & metrics & models
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    # make path for saving logs
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    # initialize logger
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)
    
    # check TensorBoard execution
    # tb = TensorBoardRunner(args.log_path, args.tb_host, args.tb_port) if args.use_tb else None

    # # define writer
    # writer = SummaryWriter(log_dir=os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'), filename_suffix=f'_{curr_time}')

    wandb.init(config=args,project='fedcola', entity='cola', name=f"{args.exp_name}{'_aux' if args.with_aux else ''}{'_attn' if args.with_aux and args.aux_attn_only else ''}{'_mlp' if args.with_aux and args.aux_mlp_only else ''}{'_'+str(args.aux_trained) if args.with_aux else ''}_{args.shared_param}_{args.share_scope}{'_comp' if args.compensation else ''}_{args.colearn_param}_{args.warmup_modality}_{args.freeze_modality}_{curr_time}")
    # run main program
    torch.autograd.set_detect_anomaly(True)
    # try:
    main(args, wandb)
    sys.exit(0)
    # except Exception:
    #     traceback.print_exc()
    #     if args.use_tb:
    #         tb.interrupt()
    #     sys.exit(1)
