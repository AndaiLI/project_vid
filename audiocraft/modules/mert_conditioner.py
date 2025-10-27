# In: audiocraft/modules/mert_conditioner.py

import typing as tp
import torch
import librosa
from .conditioners import BaseConditioner, WavCondition

class MERTStyleConditioner(BaseConditioner):
    """
    A conditioner that extracts a style vector from an audio waveform using a MERT model.
    This class is self-contained and handles its own model loading and processing.
    """
    def __init__(self, output_dim: int, model_path: str, embedding_dim: int, device: tp.Union[torch.device, str]):
        # The input dimension for the projection layer is the MERT embedding dimension.
        super().__init__(embedding_dim, output_dim)
        from transformers import AutoModel, AutoFeatureExtractor
        
        self.device = device
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, trust_remote_code=True)

    def tokenize(self, wav_condition: tp.Any):
        # For this conditioner, tokenization is a no-op.
        # The actual audio tensor is received during the forward pass.
        return wav_condition

    def forward(self, wav_condition: WavCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """

        Extracts the style vector from the provided audio waveform.
        
        Args:
            wav_condition (WavCondition): A tuple containing the audio tensor and its sample rate.
        
        Returns:
            A tuple of (embedding, mask). The embedding has a shape of [B, 1, D].
        """
        audio = wav_condition.wav.to(self.device)
        sample_rates = wav_condition.sample_rate

        # Ensure all samples in the batch have the same sample rate
        sr = sample_rates[0]
        assert all(s == sr for s in sample_rates), "All samples in a batch must have the same sample rate."

        with torch.no_grad():
            target_sr = self.feature_extractor.sampling_rate
            style_vectors = []
            
            # Process each audio in the batch
            for i in range(audio.shape[0]):
                wav_mono = audio[i].mean(dim=0).cpu().numpy()
                
                # Resample if necessary
                if sr != target_sr:
                    wav_resampled = librosa.resample(y=wav_mono, orig_sr=sr, target_sr=target_sr)
                else:
                    wav_resampled = wav_mono
                    
                inputs = self.feature_extractor(wav_resampled, sampling_rate=target_sr, return_tensors="pt").to(self.device)
                style_vector = torch.mean(self.model(**inputs).last_hidden_state, dim=1)
                style_vectors.append(style_vector)
            
            embed = torch.cat(style_vectors, dim=0)
        
        # Project the embedding to the desired output dimension and add a time step dimension.
        out_embed = self.output_proj(embed).unsqueeze(1)
        
        # The mask is always 1 as the style vector is a single global token.
        mask = torch.ones(out_embed.shape[:2], device=out_embed.device, dtype=torch.long)
        
        return out_embed, mask