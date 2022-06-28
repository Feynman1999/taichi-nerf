import logging

embedder_view = dict(
    type = "SHEncoding",
    input_dim = 3, 
    degree = 4
)

bounding_box = [[-5, -5, -5], [5, 5, 2]]

# model settings
model = dict(
    type="Coarse_Fine_Nerf",
    coarse_net = dict(
        type = 'NGP',
        hash_net = dict(
            type = "HashEncoding",
            bounding_box = bounding_box, # 空间大小
            finest_resolution = 1024,
            log2_hashmap_size = 19,
        ),
        implicit_net = dict(
            type = "MLP",
            num_layers = 2,
            hidden_dim = 64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color = 64
        ),
        embedder_view = embedder_view
    ),
    fine_net = dict(
        type = 'NGP',
        hash_net = dict(
            type = "HashEncoding",
            bounding_box = bounding_box, # 空间大小
            finest_resolution = 1024,
            log2_hashmap_size = 19,
        ),
        implicit_net = dict(
            type = "MLP",
            num_layers = 2,
            hidden_dim = 64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color = 64
        ),
        embedder_view = embedder_view
    )
)

train_cfg = dict(
    near = 0.1,
    far = 30,
    bounding_box = bounding_box,
    N_samples = 128,  # number of coarse samples per ray
    N_importance = 128,  # number of additional fine samples per ray
    lindisp=False,
    perturb=True,
    sparse_loss_weight = 1e-8,
    tv_loss_weight = 1e-8
)

test_cfg = dict(
    near = 0.1,
    far = 30,
    bounding_box = bounding_box,
    N_samples = 128,  # number of coarse samples per ray
    N_importance = 128,  # number of additional fine samples per ray
    lindisp=False,
    num_cols = 2)


root_path= r"E:\BaiduNetdiskDownload\test_a\test_a\F1_06\000060" # r"/data/test_a/F1_06/000060"
para_path = r"E:\BaiduNetdiskDownload\camera_parameters\camera_parameters" # "/home/chenyuxiang/repos/evaluation_code/camera_parameters"

log_config = dict(
    interval=50000,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

total_epochs = 5
log_level = "INFO"
work_dir = "./workdirs/ngp_tv"
load_from = None
resume_from = None
workflow = [("train", 1),]
