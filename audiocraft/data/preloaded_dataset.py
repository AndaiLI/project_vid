# 文件名: audiocraft/data/preloaded_dataset.py (最终修正版)

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
from .audio_dataset import AudioDataset, AudioMeta

logger = logging.getLogger(__name__)

# --- 1. 顶层 collate 函数 (不变) ---
def _simple_collate(batch):
    return batch[0]


# --- 2. 包装器类 (最终修正版：真实继承) ---
class PreloadedAudioDataset(AudioDataset):
    """
    一个包装类，它在初始化时将整个数据集加载到内存中。
    它通过真实继承 AudioDataset 来确保类型检查通过。
    """
    def __init__(self, underlying_dataset: AudioDataset, num_workers: int = 8):
        
        # ========== 核心修正：正确地初始化父类 ==========
        # 解释: 我们需要调用 AudioDataset.__init__。为了通过其 `assert len(meta) > 0`
        #       的检查，我们直接“借用”被包装的数据集的 `meta` 列表。
        #       同时，我们需要将所有 `underlying_dataset` 的相关属性都复制过来，
        #       以防父类方法被意外调用时出错。
        
        # a. 调用父类构造函数
        super().__init__(
            meta=underlying_dataset.meta,
            segment_duration=underlying_dataset.segment_duration,
            shuffle=underlying_dataset.shuffle,
            num_samples=underlying_dataset.num_samples,
            sample_rate=underlying_dataset.sample_rate,
            audio_encoder_sr=underlying_dataset.audio_encoder_sr,
            video_fps=underlying_dataset.video_fps,
            video_overlap=underlying_dataset.video_overlap,
            if_add_gobal=underlying_dataset.if_add_gobal,
            global_mode=underlying_dataset.global_mode,
            global_num_frames=underlying_dataset.global_num_frames,
            global_feature_path=underlying_dataset.global_feature_path,
            channels=underlying_dataset.channels,
            pad=underlying_dataset.pad,
            sample_on_duration=underlying_dataset.sample_on_duration,
            sample_on_weight=underlying_dataset.sample_on_weight,
            min_segment_ratio=underlying_dataset.min_segment_ratio,
            max_read_retry=underlying_dataset.max_read_retry,
            return_info=underlying_dataset.return_info,
            min_audio_duration=underlying_dataset.min_audio_duration,
            max_audio_duration=underlying_dataset.max_audio_duration,
            shuffle_seed=underlying_dataset.shuffle_seed,
            load_wav=underlying_dataset.load_wav,
            permutation_on_files=underlying_dataset.permutation_on_files
        )
        # ===============================================

        self.underlying_dataset = underlying_dataset
        self.data_cache = []
        
        logger.info(f"开始预加载 {len(underlying_dataset)} 个样本到内存中...")
        
        # --- 预加载逻辑 (不变) ---
        temp_loader = torch.utils.data.DataLoader(
            underlying_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=_simple_collate
        )

        for sample in tqdm(temp_loader, desc="预加载数据到RAM"):
            self.data_cache.append(sample)
            
        del temp_loader
        logger.info(f"✅ 数据预加载完成！ {len(self.data_cache)} 个样本已加载到内存。")

    def __len__(self):
        """重写 len 方法，返回缓存的大小。"""
        return len(self.data_cache)

    def __getitem__(self, index):
        """重写 getitem 方法，从缓存中读取。"""
        # 我们完全忽略父类的 __getitem__，直接返回我们缓存的数据
        return self.data_cache[index]
    
    @property
    def collater(self):
        """重写 collater，确保使用的是底层数据集的 collater。"""
        return self.underlying_dataset.collater

# --- 3. 删除虚拟子类注册 ---
# AudioDataset.register(PreloadedAudioDataset) # <--- 删除这一行