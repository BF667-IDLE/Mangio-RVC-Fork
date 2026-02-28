import sys, os
from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True" if len(sys.argv) > 5 else False
import numpy as np, traceback
from slicer2 import Slicer
import librosa
from scipy.io import wavfile
import multiprocessing
from my_utils import load_audio
import tqdm

DoFormant = False
Quefrency = 1.0
Timbre = 1.0

mutex = multiprocessing.Lock()
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
        if len(tmp_audio) == 0:  # Bug fix: Check for empty audio
            return
            
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        if tmp_max > 0:  # Bug fix: Avoid division by zero
            tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
                1 - self.alpha
            ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000, res_type='kaiser_best'  # Bug fix: Added res_type parameter
        )
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr, DoFormant, Quefrency, Timbre)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            if len(audio) > 0:  # Bug fix: Check if audio is not empty
                audio = signal.lfilter(self.bh, self.ah, audio)

                idx1 = 0
                slices = list(self.slicer.slice(audio))  # Bug fix: Convert generator to list to avoid exhaustion
                
                for audio_slice in slices:
                    i = 0
                    while True:
                        start = int(self.sr * (self.per - self.overlap) * i)
                        i += 1
                        start_idx = start
                        end_idx = start + int(self.per * self.sr)
                        
                        if len(audio_slice[start_idx:]) > self.tail * self.sr:
                            if end_idx <= len(audio_slice):  # Bug fix: Check bounds
                                tmp_audio = audio_slice[start_idx:end_idx]
                                self.norm_write(tmp_audio, idx0, idx1)
                                idx1 += 1
                        else:
                            if start_idx < len(audio_slice):  # Bug fix: Check bounds
                                tmp_audio = audio_slice[start_idx:]
                                self.norm_write(tmp_audio, idx0, idx1)
                                idx1 += 1
                            break
                println("%s->Suc." % path)
        except Exception as e:
            println("%s->%s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos, thread_n):
        for path, idx0 in tqdm.tqdm(
            infos, position=thread_n, leave=False, desc="thread:%s" % thread_n  # Bug fix: Changed leave to False
        ):
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            file_list = sorted(list(os.listdir(inp_root)))
            if not file_list:  # Bug fix: Check if directory is empty
                println("No files found in input directory")
                return
                
            infos = [
                (os.path.join(inp_root, name), idx)  # Bug fix: Use os.path.join for cross-platform compatibility
                for idx, name in enumerate(file_list)
            ]
            if noparallel:
                # Bug fix: Correct the slicing logic for noparallel mode
                chunk_size = len(infos) // n_p + (1 if len(infos) % n_p else 0)
                for i in range(n_p):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(infos))
                    if start_idx < len(infos):
                        self.pipeline_mp(infos[start_idx:end_idx], i)
            else:
                ps = []
                for i in range(n_p):
                    # Bug fix: Properly slice infos for each process
                    chunk_size = len(infos) // n_p + (1 if i < len(infos) % n_p else 0)
                    start_idx = sum(chunk_size for j in range(i))
                    end_idx = start_idx + chunk_size
                    
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[start_idx:end_idx], i)
                    )
                    ps.append(p)
                    p.start()
                
                # Bug fix: Add timeout to avoid hanging
                for i, p in enumerate(ps):
                    p.join(timeout=300)  # 5 minute timeout
                    if p.is_alive():
                        println(f"Process {i} timed out, terminating...")
                        p.terminate()
                        p.join()
        except Exception as e:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir):
    pp = PreProcess(sr, exp_dir)
    println("start preprocess")
    println(sys.argv)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")
    f.close()  # Bug fix: Close the log file


if __name__ == "__main__":
    # Bug fix: Check command line arguments
    if len(sys.argv) < 6:
        print("Usage: python trainset_preprocess_pipeline_print.py <inp_root> <sr> <n_p> <exp_dir> <noparallel>")
        sys.exit(1)
    
    preprocess_trainset(inp_root, sr, n_p, exp_dir)
