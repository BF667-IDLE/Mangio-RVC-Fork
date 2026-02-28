import sys, os
from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

# Check command line arguments first
if len(sys.argv) < 6:
    print("Usage: python trainset_preprocess_pipeline_print.py <inp_root> <sr> <n_p> <exp_dir> <noparallel>")
    print("Example: python trainset_preprocess_pipeline_print.py /content/dataset/ 40000 2 /content/logs/exp1 False")
    sys.exit(1)

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True" if len(sys.argv) > 5 else False

import numpy as np
import traceback
from slicer2 import Slicer
import librosa
from scipy.io import wavfile
import multiprocessing
from my_utils import load_audio
import tqdm

DoFormant = False
Quefrency = 1.0
Timbre = 1.0

# Fix: Use Manager for lock to work with multiprocessing
manager = multiprocessing.Manager()
mutex = manager.Lock()
f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    mutex.acquire()
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    mutex.release()


class PreProcess:
    def __init__(self, sr, exp_dir):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = 3.0
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        try:
            if len(tmp_audio) == 0:  # Fix: Check for empty audio
                println("Warning: Empty audio slice for %s_%s" % (idx0, idx1))
                return
                
            tmp_max = np.abs(tmp_audio).max()
            if tmp_max > 2.5:
                println("%s-%s-%s-filtered (amplitude too high: %s)" % (idx0, idx1, tmp_max, tmp_max))
                return
                
            if tmp_max > 0:  # Fix: Avoid division by zero
                tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
                    1 - self.alpha
                ) * tmp_audio
                
            # Write original sample rate file
            wavfile.write(
                "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
                self.sr,
                tmp_audio.astype(np.float32),
            )
            
            # Resample to 16kHz
            if len(tmp_audio) > 0:
                tmp_audio = librosa.resample(
                    tmp_audio, 
                    orig_sr=self.sr, 
                    target_sr=16000, 
                    res_type='kaiser_best'  # Fix: Added res_type parameter
                )
                wavfile.write(
                    "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
                    16000,
                    tmp_audio.astype(np.float32),
                )
        except Exception as e:
            println("Error in norm_write for %s_%s: %s" % (idx0, idx1, str(e)))

    def pipeline(self, path, idx0):
        try:
            if not os.path.exists(path):
                println("File not found: %s" % path)
                return
                
            audio = load_audio(path, self.sr, DoFormant, Quefrency, Timbre)
            
            # Fix: Check if audio is not empty
            if len(audio) == 0:
                println("Empty audio file: %s" % path)
                return
                
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            slices = list(self.slicer.slice(audio))  # Fix: Convert generator to list to avoid exhaustion
            
            for audio_slice in slices:
                if len(audio_slice) == 0:
                    continue
                    
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    start_idx = start
                    end_idx = start + int(self.per * self.sr)
                    
                    # Fix: Check bounds properly
                    if start_idx >= len(audio_slice):
                        break
                        
                    if len(audio_slice[start_idx:]) > self.tail * self.sr:
                        if end_idx <= len(audio_slice):
                            tmp_audio = audio_slice[start_idx:end_idx]
                            self.norm_write(tmp_audio, idx0, idx1)
                            idx1 += 1
                    else:
                        tmp_audio = audio_slice[start_idx:]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
                        
            println("%s->Suc. (created %s slices)" % (path, idx1))
        except Exception as e:
            println("%s->Error: %s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos, thread_n):
        try:
            for path, idx0 in tqdm.tqdm(
                infos, 
                position=thread_n, 
                leave=False, 
                desc="thread:%s" % thread_n  # Fix: Changed leave to False
            ):
                self.pipeline(path, idx0)
        except Exception as e:
            println("Thread %s error: %s" % (thread_n, str(e)))

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            # Fix: Check if input directory exists
            if not os.path.exists(inp_root):
                println("Input directory not found: %s" % inp_root)
                return
                
            # Fix: Get all audio files (support common formats)
            valid_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
            file_list = []
            for file in sorted(os.listdir(inp_root)):
                if file.lower().endswith(valid_extensions):
                    file_list.append(file)
                    
            if not file_list:
                println("No audio files found in %s" % inp_root)
                return
                
            println("Found %s audio files to process" % len(file_list))
            
            infos = [
                (os.path.join(inp_root, name), idx)  # Fix: Use os.path.join
                for idx, name in enumerate(file_list)
            ]
            
            if noparallel:
                # Fix: Sequential processing
                for i in range(n_p):
                    chunk_size = len(infos) // n_p
                    remainder = len(infos) % n_p
                    
                    start_idx = i * chunk_size + min(i, remainder)
                    end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                    
                    if start_idx < len(infos):
                        println("Processing chunk %s/%s (files %s to %s)" % 
                               (i+1, n_p, start_idx, min(end_idx, len(infos))))
                        self.pipeline_mp(infos[start_idx:end_idx], i)
            else:
                # Fix: Parallel processing with proper chunk distribution
                ps = []
                for i in range(n_p):
                    chunk_size = len(infos) // n_p
                    remainder = len(infos) % n_p
                    
                    start_idx = i * chunk_size + min(i, remainder)
                    end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                    
                    if start_idx >= len(infos):
                        continue
                        
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, 
                        args=(infos[start_idx:end_idx], i)
                    )
                    ps.append(p)
                    p.start()
                    println("Started process %s for files %s to %s" % 
                           (i, start_idx, min(end_idx, len(infos))))
                
                # Fix: Add timeout to avoid hanging
                for i, p in enumerate(ps):
                    p.join(timeout=3600)  # 1 hour timeout
                    if p.is_alive():
                        println("Process %s timed out, terminating..." % i)
                        p.terminate()
                        p.join()
                        
        except Exception as e:
            println("Pipeline error: %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir):
    try:
        pp = PreProcess(sr, exp_dir)
        println("=== Starting preprocessing ===")
        println("Input directory: %s" % inp_root)
        println("Sample rate: %s" % sr)
        println("Processes: %s" % n_p)
        println("Experiment directory: %s" % exp_dir)
        println("Parallel mode: %s" % ("Disabled" if noparallel else "Enabled"))
        println("=" * 30)
        
        pp.pipeline_mp_inp_dir(inp_root, n_p)
        
        println("=" * 30)
        println("Preprocessing completed successfully")
    except Exception as e:
        println("Preprocessing failed: %s" % traceback.format_exc())
    finally:
        f.close()  # Fix: Close the log file


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir)
