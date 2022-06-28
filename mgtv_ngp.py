import platform

if 'windows' in platform.system().lower():
    root_path= r"E:\BaiduNetdiskDownload\test_a\test_a\F1_06\000060" #
    para_path = r"E:\BaiduNetdiskDownload\camera_parameters\camera_parameters" # 

elif 'linux' in platform.system().lower():
    root_path = r"/data/test_a/F1_06/000060"
    para_path = r"/home/chenyuxiang/repos/evaluation_code/camera_parameters"


work_dir = "./workdirs/ngp"
log_level = "INFO"