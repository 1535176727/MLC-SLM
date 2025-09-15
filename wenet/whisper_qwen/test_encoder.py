# test_encoder_comparison_with_processor.py
import torch
import numpy as np
import yaml
import os
import sys

# --- 添加项目根目录到 sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")

try:
    # 移除 AutoFeatureExtractor
    # 从 WeNet 导入
    from wenet.transformer.encoder import TransformerEncoder
    from wenet.utils.checkpoint import load_whisper_encoder
    # Qwen2Audio
    from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
    # --- 导入 WeNet processor 函数 ---
    from wenet.dataset.processor import compute_log_mel_spectrogram, resample, singal_channel
except ImportError as e:
    print(f"Error importing necessary libraries: {e}")
    print("Please ensure WeNet and Transformers are correctly installed.")
    sys.exit(1)

import librosa # 仅用于生成 dummy data，如果用真实文件则不需要

def test_encoder_outputs():
    """
    比较 Whisper 和 Qwen2Audio 编码器对同一段音频（包含填充）的处理和输出。
    (加载方式基于 YAML 配置和 init_model.py, 特征计算使用 processor.py)
    """
    # --- 配置文件路径 ---
    config_path = os.path.join(project_root, "/node/myx/MLC-SLM-LLM/examples/mlcslm/asr1/conf/train_mlcslm_baseline_step1.yaml")

    # --- 加载配置 ---
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as fin:
            configs = yaml.safe_load(fin)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}"); return
    except Exception as e:
        print(f"Error loading YAML configuration: {e}"); return

    encoder_conf = configs.get('encoder_conf', {})
    whisper_checkpoint_path = configs.get('whisper_checkpoint')
    qwen_audio_model_path = configs.get('qwen_audio_path')
    input_dim = configs.get('input_dim') # 必须与特征计算结果匹配
    # --- 获取特征提取配置 ---
    dataset_conf = configs.get('dataset_conf', {})
    feature_conf = dataset_conf.get('log_mel_spectrogram_conf', {})
    resample_conf = dataset_conf.get('resample_conf', {})
    target_sample_rate = resample_conf.get('resample_rate', 16000)
    # ------------------------
    attn_implementation = None
    dtype = torch.bfloat16

    if not whisper_checkpoint_path or not qwen_audio_model_path or not input_dim:
        print("Error: Missing necessary configurations in YAML"); return

    # 路径处理：配置文件中的路径可能是相对路径，转换为相对于 conf 目录的上一级目录
    config_dir = os.path.dirname(config_path)
    base_dir = os.path.dirname(config_dir) # 获取 conf 目录的上级目录
    
    if not os.path.isabs(whisper_checkpoint_path) and whisper_checkpoint_path.startswith("./"):
        whisper_checkpoint_path = os.path.normpath(os.path.join(base_dir, whisper_checkpoint_path))
    elif not os.path.isabs(whisper_checkpoint_path):
        # If relative but doesn't start with ./, maybe it's relative to project root or something else?
        # Assuming relative to base_dir for now based on the issue.
         whisper_checkpoint_path = os.path.normpath(os.path.join(base_dir, whisper_checkpoint_path))
        
    if not os.path.isabs(qwen_audio_model_path) and qwen_audio_model_path.startswith("./"):
        qwen_audio_model_path = os.path.normpath(os.path.join(base_dir, qwen_audio_model_path))
    elif not os.path.isabs(qwen_audio_model_path):
         qwen_audio_model_path = os.path.normpath(os.path.join(base_dir, qwen_audio_model_path))

    print(f"Whisper Checkpoint: {whisper_checkpoint_path}")
    print(f"Qwen Audio Path: {qwen_audio_model_path}")
    print(f"Target Sample Rate: {target_sample_rate}")
    print(f"Feature Config: {feature_conf}")
    print(f"Input Dim (num_mel_bins): {input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 切换回 GPU
    # device = torch.device("cpu") # 强制使用 CPU
    print(f"Using device: {device}, dtype: {dtype}")

    # --- 加载模型 --- (不再加载 FeatureExtractor)
    print("Loading models...")
    try:
        # 加载 Whisper 编码器
        print("Initializing WeNet TransformerEncoder for Whisper...")
        if 'output_size' not in encoder_conf:
             print("Warning: 'output_size' not in encoder_conf, setting to 1280 for whisper-large-v3.")
             encoder_conf['output_size'] = 1280

        whisper_encoder = TransformerEncoder(
            input_size=input_dim,
            **encoder_conf
        ).to(device).to(dtype) # 移动到 GPU

        print(f"Loading Whisper weights directly from checkpoint: {whisper_checkpoint_path}")
        try:
            # 加载权重时 map_location 不再强制为 cpu
            checkpoint = torch.load(whisper_checkpoint_path, map_location=device)
            checkpoint_encoder = {'.'.join(k.split('.')[1:]): v for k, v in checkpoint.items() if 'encoder' in k}
            missing_keys, unexpected_keys = whisper_encoder.load_state_dict(checkpoint_encoder, strict=False)
            if missing_keys: print(f"Warning: Missing Whisper keys: {missing_keys}")
            if unexpected_keys: print(f"Warning: Unexpected Whisper keys: {unexpected_keys}")
            print("Whisper encoder weights loaded.")
        except FileNotFoundError: print(f"Error: Whisper checkpoint not found at {whisper_checkpoint_path}"); return
        except Exception as e: print(f"Error loading Whisper weights: {e}"); import traceback; traceback.print_exc(); return
        whisper_encoder.eval()

        # 加载 Qwen2Audio 编码器
        print(f"Loading Qwen2Audio model from: {qwen_audio_model_path}...")
        qwen2_audio_full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            qwen_audio_model_path,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device) # 移动到 GPU
        qwen_encoder = qwen2_audio_full_model.audio_tower # 您上次修改为了 audio_tower
        qwen_encoder.eval()

    except Exception as e:
        import traceback
        print(f"Error loading models: {e}"); traceback.print_exc(); return

    # --- 创建音频数据 ---
    print("Creating dummy audio data...")
    sampling_rate = target_sample_rate
    duration_seconds = 5
    dummy_waveform_np = np.sin(2 * np.pi * 440.0 * np.arange(sampling_rate * duration_seconds) / sampling_rate).astype(np.float32)
    # 张量移动到 GPU
    dummy_waveform = torch.from_numpy(dummy_waveform_np).unsqueeze(0).to(device)
    print(f"Dummy waveform shape: {dummy_waveform.shape}, Sampling rate: {sampling_rate}")

    # --- 使用 processor 计算特征 ---
    print("Calculating Mel features using wenet.dataset.processor...")
    sample_dict = {
        'key': 'dummy',
        'wav': dummy_waveform, # GPU tensor
        'sample_rate': sampling_rate,
    }
    try:
        processed_sample = compute_log_mel_spectrogram(sample_dict, **feature_conf)
        # 特征移到 GPU
        input_features_single = processed_sample['feat'].to(device).to(dtype)

        input_features = input_features_single.unsqueeze(0).transpose(1, 2)
        B, D_mel, T_mel = input_features.shape

    except Exception as e:
        print(f"Error calculating features using processor: {e}")
        import traceback; traceback.print_exc(); return

    print(f"Calculated Mel features shape: {input_features.shape}")

    if D_mel != input_dim:
        print(f"Error: Calculated feature dim ({D_mel}) != config input_dim ({input_dim}).")
        return

    # --- 准备 Qwen2Audio 输入 ---
    target_len_qwen_in = 3000
    # 张量创建在 GPU 上
    qwen_input_features = torch.zeros(B, D_mel, target_len_qwen_in, device=device, dtype=dtype)
    valid_len_in = min(T_mel, target_len_qwen_in)
    qwen_input_features[:, :, :valid_len_in] = input_features[:, :, :valid_len_in]
    print(f"Qwen input features shape (padded): {qwen_input_features.shape}")

    # --- 运行编码器 ---
    print("\n--- Running Whisper Encoder (WeNet TransformerEncoder) ---")
    with torch.no_grad():
        try:
            # 张量在 GPU 上
            xs_lens = torch.tensor([T_mel] * B, device=device, dtype=torch.long)
            whisper_encoder_out, whisper_encoder_mask = whisper_encoder(input_features.transpose(1, 2), xs_lens)

            T_whisper_out = whisper_encoder_out.shape[1]
            print(f"Whisper Encoder output shape: {whisper_encoder_out.shape}")

            # 估算 Whisper 有效输出长度
            whisper_downsample = 4 # 再次确认这个假设
            T_whisper_valid_out = (xs_lens + whisper_downsample - 1) // whisper_downsample
            T_whisper_valid_out = T_whisper_valid_out[0].item()
            print(f"Estimated Whisper valid output length: {T_whisper_valid_out} (based on {T_mel} input frames, assumed downsample={whisper_downsample})")
            T_whisper_valid_out = min(T_whisper_valid_out, T_whisper_out)

            print("\nWhisper Output Values (End of valid part):")
            if T_whisper_valid_out > 0:
                 print(whisper_encoder_out[0, T_whisper_valid_out-3:T_whisper_valid_out, :5])
            else:
                 print("Valid output length is 0 or negative.")

        except Exception as e:
            import traceback; print(f"Error running Whisper encoder: {e}"); traceback.print_exc(); whisper_encoder_out = None

    print("\n--- Running Qwen2Audio Encoder ---")
    with torch.no_grad():
        try:
            # Qwen2Audio Encoder forward
            qwen_outputs = qwen_encoder(qwen_input_features, return_dict=True)
            qwen_last_hidden_state = qwen_outputs.last_hidden_state
            T_qwen_out = qwen_last_hidden_state.shape[1]
            print(f"Qwen2Audio Encoder output shape: {qwen_last_hidden_state.shape}")

            # Qwen 输出中对应于 Whisper 有效部分结束的位置
            print(f"\nQwen Output Values (at Whisper's estimated valid end index ~{T_whisper_valid_out}):")
            if T_whisper_valid_out <= T_qwen_out and T_whisper_valid_out > 0:
                 print(qwen_last_hidden_state[0, T_whisper_valid_out-3:T_whisper_valid_out, :5])
            else:
                 print(f"Index {T_whisper_valid_out} out of bounds for Qwen output (length {T_qwen_out}) or invalid.")

            # 查看 Qwen 输出的最后几个时间步 (对应于输入的填充部分)
            print("\nQwen Output Values (End of sequence, corresponding to padding):")
            print(qwen_last_hidden_state[0, -3:, :5])

        except Exception as e:
            import traceback; print(f"Error running Qwen2Audio encoder: {e}"); traceback.print_exc(); qwen_last_hidden_state = None

    print("\n--- Comparison Summary ---")
    if 'whisper_encoder_out' in locals() and whisper_encoder_out is not None:
        print(f"Whisper output sequence length: {T_whisper_out} (Valid part estimated: {T_whisper_valid_out})")
    if 'qwen_last_hidden_state' in locals() and qwen_last_hidden_state is not None:
        print(f"Qwen output sequence length: {T_qwen_out} (Fixed)")
        print("Observe the values at the end of the Qwen output - are they zeros, repeated patterns, or complex values?")

if __name__ == "__main__":
    test_encoder_outputs()
