"""Inference configuration for RDiff_seq RNA inverse design."""

import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()

    # Misc config
    config.exp_type = 'vpsde'
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.seed = 42
    config.save = True

    # Checkpoint-compatible data schema. Public release uses only inference inputs.
    config.data = data = ml_collections.ConfigDict()

    data.train_root = './'
    data.train_list = './'
    data.aug_root = './'
    data.aug_list = './'
    data.aug_root2 = './'
    data.aug_list2 = './'
    data.val_root = './'
    data.val_list = './'
    data.test_root = './'
    data.test_list = './'

    data.cluster_path = './'

    data.aug_train = True
    data.seq_centered = True
    data.use_aug = True
    data.ss_channel = 0
    data.use_nc_detail_cond = False
    data.nc_sparse_dirname = 'NC_sparse'
    data.nc_channel = 11
    data.radius = 4.5
    data.top_k = 10
    data.num_rbf = 16
    data.num_posenc = 16
    data.num_conformers = 1
    data.add_noise = 0.1
    data.knn_num = 10

    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'  # 'linear', 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    ## set smaller for faster eval
    sampling.steps = 100

    # Model config
    config.model = model = ml_collections.ConfigDict()
    model.geometric_data_parallel = False
    model.ema_decay = 0.999
    model.pred_data = True
    model.self_cond = True
    model.name = 'GVPTransCond'
    model.node_in_dim = (8, 4)
    model.node_h_dim = (512, 128)
    model.base_edge_scalar_dim = 32
    model.nc_embedding_dim = 16
    model.edge_in_dim = (model.base_edge_scalar_dim, 1)
    model.edge_h_dim = (128, 1)
    model.num_layers = 4
    model.drop_rate = 0.1
    model.out_dim = 4
    model.time_cond = True
    model.dihedral_angle = True
    model.num_trans_layer = 8
    model.drop_struct = 0.4
    model.ss_emb = False

    model.trans = trans = ml_collections.ConfigDict()
    trans.encoder_embed_dim = 512
    trans.encoder_attention_heads = 16
    trans.attention_dropout = 0.1
    trans.dropout = 0.1
    trans.encoder_ffn_embed_dim = 1024

    # Checkpoint-compatible training schema. Training dataloaders are not included.
    config.train = train = ml_collections.ConfigDict()
    train.reduce_mean = False #True
    train.epochs = 60 # 50
    train.batch_size = 8 # 16
    train.num_workers = 4 # 16
    train.val_freq = 30000  # validate every step
    train.save_freq = 100000  # save model every step
    train.print_freq = 1000  # print training info every step
    train.sample_mode = 'cluster_time_batch'  # 'time_batch', 'cluster_time_batch'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 20000
    optim.grad_clip = 20.
    optim.disable_grad_log = True

    # Evaluation config
    config.eval = eval = ml_collections.ConfigDict()
    eval.model_path = ''
    eval.input_ss_npy = ''
    eval.input_nc_npz = ''
    eval.input_db = ''
    eval.output_fasta_file = ''
    eval.test_perplexity = False
    eval.test_recovery = True
    eval.n_samples = 1
    eval.sampling_steps = 100
    eval.cond_scale = -1.
    eval.dynamic_threshold = False
    eval.dynamic_thresholding_percentile = 0.95
    eval.tradeoff_n_seqs = 20
    eval.batch_size = 50

    ## fixed sequence motifs
    eval.fixed_part = False
    eval.fixed_seq = ''
    eval.fixed_seq_path = ''

    ## partial sampling
    eval.partial_sampling = False
    eval.target_seq = ''
    eval.partial_min_t = 0.15
    eval.partial_max_t = 0.55

    return config
