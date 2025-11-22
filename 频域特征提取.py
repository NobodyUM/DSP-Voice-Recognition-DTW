import numpy as np
import scipy.io.wavfile
from scipy import signal
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import scipy.io.wavfile

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


def endpoint_detection(signal_wave, sample_rate):
    """
    端点检测函数：通过改进的双门限法检测语音有效区间
    
    参数:
    signal_wave: 原始语音信号
    sample_rate: 采样率
    
    返回:
    valid_signal: 有效区间对应的原始语音段
    start_index: 有效区间起始索引
    end_index: 有效区间结束索引
    """
    
    # 参数设置
    frame_length = int(0.025 * sample_rate)  # 25ms
    frame_shift = int(0.01 * sample_rate)    # 10ms

    # 1. 分帧
    signal_length = len(signal_wave)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_shift)) + 1
    pad_length = (num_frames - 1) * frame_shift + frame_length
    
    if pad_length > signal_length:
        padding = np.zeros(pad_length - signal_length)
        padded_signal = np.concatenate((signal_wave, padding))
    else:
        padded_signal = signal_wave
    
    frames = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = padded_signal[start:end]
        frames.append(frame)
    frames = np.array(frames)
    
    # 2. 加窗
    hamming_window = np.hamming(frame_length)
    windowed_frames = frames * hamming_window
    
    # 3. 计算短时能量
    frame_energies = np.sum(windowed_frames ** 2, axis=1)
    
    # 4. 计算短时过零率
    def zero_crossing_rate(frame):
        """计算一帧信号的过零率"""
        signs = np.sign(frame)
        crossings = np.abs(np.diff(signs)) / 2
        return np.sum(crossings)
    
    zcr = np.array([zero_crossing_rate(frame) for frame in windowed_frames])
    
    # 5. 改进的双门限法端点检测
    # 计算能量和过零率的统计特征
    energy_mean = np.mean(frame_energies)
    energy_std = np.std(frame_energies)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # 设置门限
    energy_high_threshold = energy_mean + 0.85 * energy_std  # 提高高门限
    energy_low_threshold = energy_mean + 0.15 * energy_std   # 降低低门限
    zcr_threshold = zcr_mean + 0.8 * zcr_std               # 提高过零率门限
    
    # 端点检测状态机 - 改进版本
    status = 0  # 0:静音, 1:可能开始, 2:语音段, 3:可能结束
    speech_start = 0
    speech_end = 0
    min_speech_length = 10  # 最小语音长度（帧）
    max_silence_gap = 20    # 最大允许的静音间隔（帧）
    
    # 第一遍：检测可能的语音段
    speech_segments = []
    current_start = -1
    silence_counter = 0
    
    for i in range(len(frame_energies)):
        if status == 0:  # 静音状态
            if frame_energies[i] > energy_high_threshold or zcr[i] > zcr_threshold:
                status = 1
                current_start = i
                silence_counter = 0
        elif status == 1:  # 可能开始
            if frame_energies[i] > energy_high_threshold or zcr[i] > zcr_threshold:
                status = 2
                silence_counter = 0
            elif frame_energies[i] < energy_low_threshold and zcr[i] < zcr_threshold:
                silence_counter += 1
                if silence_counter > 5:  # 连续5帧静音则放弃
                    status = 0
                    current_start = -1
                    silence_counter = 0
        elif status == 2:  # 语音段
            if frame_energies[i] < energy_low_threshold and zcr[i] < zcr_threshold:
                status = 3
                silence_counter = 1
            else:
                silence_counter = 0
        elif status == 3:  # 可能结束
            if frame_energies[i] > energy_high_threshold or zcr[i] > zcr_threshold:
                # 重新检测到语音，回到语音段状态
                status = 2
                silence_counter = 0
            else:
                silence_counter += 1
                if silence_counter > max_silence_gap:  # 连续静音超过阈值，确认结束
                    # 检查语音段长度
                    if i - current_start >= min_speech_length:
                        speech_segments.append((current_start, i))
                    status = 0
                    current_start = -1
                    silence_counter = 0
    
    # 处理最后一个语音段
    if status in [2, 3] and current_start != -1 and (len(frame_energies) - 1 - current_start) >= min_speech_length:
        speech_segments.append((current_start, len(frame_energies) - 1))
    
    # 6. 合并相邻的语音段
    merged_segments = []
    if speech_segments:
        merged_segments.append(speech_segments[0])
        for i in range(1, len(speech_segments)):
            last_segment = merged_segments[-1]
            current_segment = speech_segments[i]
            
            # 如果两个段之间的距离小于阈值，则合并
            if current_segment[0] - last_segment[1] < max_silence_gap:
                merged_segments[-1] = (last_segment[0], current_segment[1])
            else:
                merged_segments.append(current_segment)
    
    # 7. 提取最长的语音段或所有语音段
    if merged_segments:
        # 选择最长的语音段
        longest_segment = max(merged_segments, key=lambda x: x[1] - x[0])
        start_frame, end_frame = longest_segment
        
        # 转换为原始信号的索引
        start_index = start_frame * frame_shift
        end_index = min(end_frame * frame_shift + frame_length, len(signal_wave))
        
        # 提取有效区间的原始语音段（预加重之前的）
        valid_signal = signal_wave[start_index:end_index]
        
        print(f"端点检测结果: 起始帧 {start_frame}, 结束帧 {end_frame}")
        print(f"原始信号索引: {start_index} - {end_index}")
        print(f"有效语音长度: {len(valid_signal)} 采样点 ({len(valid_signal)/sample_rate:.2f} 秒)")
        
        # 可视化端点检测过程
        # visualize_endpoint_detection(signal_wave, sample_rate, frame_energies, zcr, 
        #                            energy_high_threshold, energy_low_threshold, zcr_threshold,
        #                            start_frame, end_frame, frame_shift, frame_length)
        
        return valid_signal, start_index, end_index
    else:
        print("未检测到有效语音段")
        return signal_wave, 0, len(signal_wave)


def visualize_endpoint_detection(signal_wave, sample_rate, frame_energies, zcr,
                               energy_high_threshold, energy_low_threshold, zcr_threshold,
                               start_frame, end_frame, frame_shift, frame_length):
    """可视化端点检测过程"""
    plt.figure(figsize=(12, 8))
    
    # 1. 原始信号和检测结果
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(signal_wave)) / sample_rate
    plt.plot(time_axis, signal_wave)
    
    # 标记有效区间
    start_time = start_frame * frame_shift / sample_rate
    end_time = min(end_frame * frame_shift + frame_length, len(signal_wave)) / sample_rate
    plt.axvspan(start_time, end_time, alpha=0.3, color='red', label='有效区间')
    
    plt.title('原始语音信号及检测的有效区间')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 2. 短时能量和门限
    plt.subplot(3, 1, 2)
    frame_time = np.arange(len(frame_energies)) * frame_shift / sample_rate
    plt.plot(frame_time, frame_energies, label='短时能量')
    plt.axhline(y=energy_high_threshold, color='r', linestyle='--', label='高能量门限')
    plt.axhline(y=energy_low_threshold, color='orange', linestyle='--', label='低能量门限')
    
    # 标记有效区间
    start_time_frame = start_frame * frame_shift / sample_rate
    end_time_frame = end_frame * frame_shift / sample_rate
    plt.axvspan(start_time_frame, end_time_frame, alpha=0.3, color='red')
    
    plt.title('短时能量及门限')
    plt.xlabel('时间 (s)')
    plt.ylabel('能量')
    plt.legend()
    plt.grid(True)
    
    # 3. 过零率和门限
    plt.subplot(3, 1, 3)
    plt.plot(frame_time, zcr, label='过零率')
    plt.axhline(y=zcr_threshold, color='g', linestyle='--', label='过零率门限')
    
    # 标记有效区间
    plt.axvspan(start_time_frame, end_time_frame, alpha=0.3, color='red')
    
    plt.title('过零率及门限')
    plt.xlabel('时间 (s)')
    plt.ylabel('过零率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def extract_features_from_valid_segment(valid_signal, sample_rate, include_delta=True, include_delta_delta=True):
    """
    在有效语音区间内提取特征
    
    参数:
    valid_signal: 有效区间的语音信号
    sample_rate: 采样率
    include_delta: 是否包含一阶差分特征
    include_delta_delta: 是否包含二阶差分特征
    
    返回:
    features: 特征字典，包含多种语音特征
    """
    
    # 参数设置
    frame_length = 512
    frame_shift = 200
    n_fft = 512
    nfilt = 40
    num_ceps = 13
    
    # 1. 预加重
    pre_emphasis = 0.97
    emphasized_signal = np.append(valid_signal[0], valid_signal[1:] - pre_emphasis * valid_signal[:-1])
    
    # 2. 分帧
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_shift)) + 1
    pad_length = (num_frames - 1) * frame_shift + frame_length
    
    if pad_length > signal_length:
        padding = np.zeros(pad_length - signal_length)
        padded_signal = np.concatenate((emphasized_signal, padding))
    else:
        padded_signal = emphasized_signal
    
    frames = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = padded_signal[start:end]
        frames.append(frame)
    frames = np.array(frames)
    
    # 3. 加窗
    hamming_window = np.hamming(frame_length)
    windowed_frames = frames * hamming_window
    
    # 4. 计算频谱特征
    magnitude_frames = np.zeros((num_frames, n_fft//2 + 1))
    energy_frames = np.zeros((num_frames, n_fft//2 + 1))
    
    for i in range(num_frames):
        windowed_frame = windowed_frames[i, :]
        wf_fft = np.abs(np.fft.fft(windowed_frame, n_fft))
        magnitude_frames[i, :] = wf_fft[:n_fft//2 + 1]
        energy_frames[i, :] = magnitude_frames[i, :] ** 2
    
    # 5. 计算帧能量（时域）
    frame_energies = np.sum(frames ** 2, axis=1)
    
    # 6. 计算过零率
    def zero_crossing_rate(frame):
        """计算一帧信号的过零率"""
        signs = np.sign(frame)
        crossings = np.abs(np.diff(signs)) / 2
        return np.sum(crossings) / len(frame)
    
    zcr = np.array([zero_crossing_rate(frame) for frame in frames])
    
    # 7. 计算梅尔滤波器组和MFCC
    def get_filter_banks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
        if highfreq is None:
            highfreq = samplerate / 2
        lowmel = 2595.0 * np.log10(1 + lowfreq / 700.0)
        highmel = 2595.0 * np.log10(1 + highfreq / 700.0)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        hzpoints = 700 * (10**(melpoints / 2595.0) - 1.0)
        bin = np.floor((nfft + 1) * hzpoints / samplerate)
        fbank = np.zeros([nfilt, nfft//2 + 1])
        for j in range(nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j, i] = (i - bin[j]) / (bin[j+1] - bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j, i] = (bin[j+2] - i) / (bin[j+2] - bin[j+1])
        return fbank, hzpoints
    
    filter_banks, hz_points = get_filter_banks(nfilt=nfilt, nfft=n_fft, samplerate=sample_rate)
    filter_bank_energies = np.dot(energy_frames, filter_banks.T)
    filter_bank_energies = np.where(filter_bank_energies == 0, np.finfo(float).eps, filter_bank_energies)
    log_mel_spectrogram = 10 * np.log10(filter_bank_energies)
    
    # 8. 计算MFCC
    mfcc = dct(log_mel_spectrogram, type=2, axis=1, norm='ortho')[:, :num_ceps]
    
    # # 倒谱提升，用了这个反而会降低预测准确率
    # cepstral_lifter = 22
    # (nframes, ncoeff) = mfcc.shape
    # lift = 1 + (cepstral_lifter / 2) * np.sin(np.pi * np.arange(ncoeff) / cepstral_lifter)
    # mfcc *= lift
    
    # 9. 计算一阶差分（Delta特征）
    def compute_delta(features, N=2):
        """计算特征的一阶差分"""
        delta = np.zeros_like(features)
        for t in range(len(features)):
            if t < N:
                delta[t] = features[t+1] - features[t]
            elif t >= len(features) - N:
                delta[t] = features[t] - features[t-1]
            else:
                numerator = sum(i * (features[t+i] - features[t-i]) for i in range(1, N+1))
                denominator = 2 * sum(i**2 for i in range(1, N+1))
                delta[t] = numerator / denominator
        return delta
    
    # 10. 计算二阶差分（Delta-Delta特征）
    mfcc_delta = None
    mfcc_delta_delta = None
    
    if include_delta:
        mfcc_delta = np.array([compute_delta(mfcc[:, i]) for i in range(mfcc.shape[1])]).T
    
    if include_delta_delta and include_delta:
        mfcc_delta_delta = np.array([compute_delta(mfcc_delta[:, i]) for i in range(mfcc_delta.shape[1])]).T
    
    # 11. 组合特征
    features = {}
    features['mfcc'] = mfcc  # 13维MFCC系数
    
    if include_delta:
        features['delta'] = mfcc_delta  # 13维一阶差分
    if include_delta_delta:
        features['delta_delta'] = mfcc_delta_delta  # 13维二阶差分
    
    features['log_mel_spectrogram'] = log_mel_spectrogram  # 对数梅尔频谱
    features['frame_energy'] = frame_energies  # 帧能量
    features['zcr'] = zcr  # 过零率
    features['magnitude_spectrum'] = magnitude_frames  # 幅度谱
    features['energy_spectrum'] = energy_frames  # 能量谱
    features['filter_banks'] = filter_banks  # 梅尔滤波器组
    features['hz_points'] = hz_points  # 梅尔滤波器对应的频率点
    
    # 12. 组合所有特征（用于机器学习模型）
    if include_delta and include_delta_delta:
        combined_features = np.hstack([mfcc, mfcc_delta, mfcc_delta_delta])
    elif include_delta:
        combined_features = np.hstack([mfcc, mfcc_delta])
    else:
        combined_features = mfcc
    
    features['combined_features'] = combined_features
    
    # 13. 添加元数据
    features['metadata'] = {
        'num_frames': num_frames,
        'frame_length': frame_length,
        'frame_shift': frame_shift,
        'sample_rate': sample_rate,
        'signal_length': signal_length,
        'feature_dimensions': combined_features.shape[1] if 'combined_features' in features else mfcc.shape[1]
    }
    
    return features,emphasized_signal


# 使用示例
def demonstrate_complete_pipeline():
    """演示完整的语音处理流程"""
    
    # 读取音频文件
    sample_rate, signal_wave = scipy.io.wavfile.read("C:\\Users\\24821\\Desktop\\dsp实验\\voice_set\\0\\recording_01_20251121_220811.wav")
    
    print("=== 语音端点检测 ===")
    # 1. 端点检测并提取有效区间
    valid_signal, start_idx, end_idx = endpoint_detection(signal_wave, sample_rate)
    
    print("\n=== 有效区间特征提取 ===")
    # 2. 在有效区间内提取特征
    features,emphasized_signal = extract_features_from_valid_segment(valid_signal, sample_rate, 
                                                  include_delta=True, 
                                                  include_delta_delta=True)
    
    print(f"总帧数: {features['metadata']['num_frames']}")
    print(f"特征维度: {features['metadata']['feature_dimensions']}")
    print(f"MFCC形状: {features['mfcc'].shape}")
    print(f"对数梅尔频谱形状: {features['log_mel_spectrogram'].shape}")
    # print(f"特征标题: {list(features.keys())}")
    
    # 可视化结果
    # visualize_features(features, emphasized_signal, sample_rate, valid_signal, start_idx, end_idx, signal_wave)
    
    return features, valid_signal, start_idx, end_idx


def extract_mfcc_for_training(file_path, enable_endpoint_detection=True, 
                             include_delta=True, include_delta_delta=True):
    sample_rate, signal_wave = scipy.io.wavfile.read(file_path)   
    if enable_endpoint_detection: # 统一端点检测策略
        valid_signal, start_idx, end_idx = endpoint_detection(signal_wave, sample_rate)
    else:
        valid_signal = signal_wave

    features, emphasized_signal = extract_features_from_valid_segment(
        valid_signal, sample_rate, 
        include_delta=include_delta, 
        include_delta_delta=include_delta_delta
    )
    return features['combined_features']


def visualize_features(features, emphasized_signal, sample_rate, valid_signal, start_idx, end_idx, original_signal):
    """可视化特征提取结果"""
    plt.figure(figsize=(18, 12))
    
    # 1. 原始信号和有效区间
    plt.subplot(3, 3, 1)
    time_axis = np.arange(len(original_signal)) / sample_rate
    plt.plot(time_axis, original_signal)
    plt.axvspan(start_idx/sample_rate, end_idx/sample_rate, alpha=0.3, color='red', label='有效区间')
    plt.title('原始语音信号及检测的有效区间')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    # 2. 有效区间信号
    plt.subplot(3, 3, 2)
    valid_time = np.arange(len(valid_signal)) / sample_rate
    plt.plot(valid_time, emphasized_signal)
    plt.title('有效区间预加重后的语音信号')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.grid(True)
    
    # 3. MFCC特征
    plt.subplot(3, 3, 3)
    plt.imshow(features['mfcc'].T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='MFCC系数值')
    plt.title('MFCC特征')
    plt.xlabel('帧索引')
    plt.ylabel('MFCC系数')
    
    # 4. 对数梅尔频谱
    plt.subplot(3, 3, 4)
    plt.imshow(features['log_mel_spectrogram'].T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='能量 (dB)')
    plt.title('对数梅尔频谱')
    plt.xlabel('帧索引')
    plt.ylabel('梅尔滤波器')
    
    # 5. 帧能量
    plt.subplot(3, 3, 5)
    plt.plot(features['frame_energy'])
    plt.title('帧能量')
    plt.xlabel('帧索引')
    plt.ylabel('能量')
    plt.grid(True)
    
    # 6. 过零率
    plt.subplot(3, 3, 6)
    plt.plot(features['zcr'])
    plt.title('过零率 (ZCR)')
    plt.xlabel('帧索引')
    plt.ylabel('过零率')
    plt.grid(True)
    
    # 7. 梅尔滤波器
    plt.subplot(3, 3, 7)
    for i in range(features['filter_banks'].shape[0]):
        plt.plot(features['filter_banks'][i])
    plt.title('梅尔滤波器组')
    plt.xlabel('频率点')
    plt.ylabel('幅度')
    plt.grid(True)
    
    # 8. 前5个维度组合特征
    plt.subplot(3, 3, 8)
    if 'combined_features' in features:
        # 显示组合特征的前5个维度
        for i in range(min(5, features['combined_features'].shape[1])):
            plt.plot(features['combined_features'][:, i] + i*2, label=f'特征{i+1}')
        plt.title('组合特征（前5个维度，偏移显示）')
        plt.xlabel('帧索引')
        plt.ylabel('特征值')
        plt.legend()
        plt.grid(True)
    
    # 9. 特征统计
    plt.subplot(3, 3, 9)
    feature_means = np.mean(features['mfcc'], axis=0)
    feature_stds = np.std(features['mfcc'], axis=0)
    plt.bar(range(len(feature_means)), feature_means, yerr=feature_stds, alpha=0.7)
    plt.title('MFCC系数均值和标准差')
    plt.xlabel('MFCC系数索引')
    plt.ylabel('均值±标准差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行完整流程
# print("开始完整语音处理流程...")
# result_features, valid_signal, start_idx, end_idx = demonstrate_complete_pipeline()
# print("处理完成！")