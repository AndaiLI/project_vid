# 文件名: create_finetune_checkpoint.py (修正版)

import torch
from omegaconf import OmegaConf
from audiocraft.models import builders as model_builders
from audiocraft.solvers import builders as solver_builders
import sys

print("--- 开始创建微调初始检查点 ---")

# --- 1. 加载并合并配置文件 ---
#    这是模拟 `dora run` 行为的关键步骤
print("--> 正在加载并合并配置文件...")
try:
    # a. 加载我们的主配置文件
    main_config_path = '/data/lw/musci_generation/VidMuse/VidMuse_v1/project_vid/config/solver/VidMuse/finetune.yaml' 
    cfg = OmegaConf.load(main_config_path)

    # b. 手动处理 defaults，加载并合并父配置
    #    我们需要找到 audiocraft 的根目录来正确解析路径
    #    这里简化处理，假设路径是相对于当前目录的
    #    注意：这个列表的顺序很重要，后面的会覆盖前面的
    configs_to_merge = [
        '/data/lw/musci_generation/VidMuse/VidMuse_v1/config/VidMuse/default.yaml',
        '/data/lw/musci_generation/VidMuse/VidMuse_v1/config/model/lm/musicgen_lm.yaml',
        '/data/lw/musci_generation/VidMuse/VidMuse_v1/config/dset/audio/default.yaml',
        main_config_path  # 我们自己的配置放在最后，以确保它的设置优先级最高
    ]
    
    # OmegaConf.merge 接受多个配置对象
    merged_cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in configs_to_merge])
    
    # 将合并后的配置赋给 cfg
    cfg = merged_cfg
    print("--> 配置文件合并成功。")

except FileNotFoundError as e:
    print(f"!!! 错误: 找不到配置文件 {e.filename}。请确保脚本从项目根目录运行，且路径正确。")
    sys.exit(1)
except Exception as e:
    print(f"!!! 错误: 加载或合并配置时出错: {e}")
    sys.exit(1)


# --- 2. 构建我们新的 LMModel 完整架构 ---
print("--> 正在构建新的 LMModel 架构...")
# 现在 cfg 是一个完整的配置对象，get_lm_model 可以正常工作了
# get_lm_model 需要的是包含了所有模型相关配置的顶层 cfg
model = model_builders.get_lm_model(cfg) 
print("--> 新模型架构构建成功。")

# --- 3. 加载您的 VidMuse 纯权重文件 ---
vidmuse_weights_path = '/data/lw/musci_generation/VidMuse/VidMuse_v1/project_vid/model/state_dict.bin'
print(f"--> 正在从 '{vidmuse_weights_path}' 加载预训练权重...")
vidmuse_state_dict = torch.load(vidmuse_weights_path, map_location='cpu')

# --- 4. 使用 strict=False 加载权重到新模型中 ---
missing_keys, unexpected_keys = model.load_state_dict(vidmuse_state_dict, strict=False)
print("--> 权重加载完成 (strict=False)。")
# 打印一些缺失的键，以验证我们的新模块没有被加载权重
print(f"    - 部分缺失的键 (我们的新模块): {missing_keys[:5]}...") 
print(f"    - 意外的键 (应该为空): {unexpected_keys}")

# --- 5. 创建一个全新的优化器 ---
print("--> 正在创建新的优化器...")
optimizer = solver_builders.get_optimizer(model, cfg.optim)

# --- 6. 构建一个合规的检查点字典 ---
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_state': {
        'model': model.state_dict()
    },
    'epoch': 0,
    '__version__': '1.0.0', # 添加一个版本号，更像真实的检查点
    # Dora/Flashy 需要 xp 键，即使是空的
    'xp.cfg': OmegaConf.to_yaml(cfg),
}
print("--> 合规的检查点字典已创建。")

# --- 7. 保存这个完美的初始检查点 ---
output_path = 'initial_finetune_checkpoint.th'
torch.save(checkpoint, output_path)
print(f"✅ 成功！初始检查点已保存到 '{output_path}'")
print(f"\n下一步：请在您的训练脚本中，将 continue_from 指向这个新文件的绝对路径。")