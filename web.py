import os, shutil
import sys, json
import math, signal
import traceback, warnings
import logging, threading
import csv, re
from random import shuffle
from subprocess import Popen
from time import sleep
from pathlib import Path

import numpy as np
import torch, faiss
import ffmpeg
import gradio as gr
import soundfile as sf
import scipy.io.wavfile as wavfile

from configs.config import Config
from rvc.lib.embedders.fairseq import load_model
from i18n.i18n import I18nAuto
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from uvr.infer_uvr5 import _audio_pre_, _audio_pre_new
from uvr.MDXNet import MDXNetDereverb
from rvc.infer.my_utils import load_audio, CSVutil
from rvc.train.process_ckpt import change_info, extract_small_model, merge, show_info
from rvc.infer.pipeline import VC
from sklearn.cluster import MiniBatchKMeans

# Setup directories
now_dir = os.getcwd()
sys.path.append(now_dir)

# Environment setup
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

# Create necessary directories
TEMP_DIR = os.path.join(now_dir, "TEMP")
shutil.rmtree(TEMP_DIR, ignore_errors=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "audios"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = TEMP_DIR

warnings.filterwarnings("ignore")
torch.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)

# CSV directories
CSV_DIR = "csvdb"
os.makedirs(CSV_DIR, exist_ok=True)
for csv_file in ["formanting.csv", "stop.csv"]:
    csv_path = os.path.join(CSV_DIR, csv_file)
    if not os.path.exists(csv_path):
        open(csv_path, "w").close()

# Global variables
DoFormant = False
Quefrency = 1.0
Timbre = 1.0
hubert_model = None
p = None
PID = None
cpt = None
n_spk = None
tgt_sr = None
net_g = None
vc = None
version = None
isinterrupted = 0

# Load formant settings
try:
    DoFormant, Quefrency, Timbre = CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "r", "formanting")
    DoFormant = DoFormant.lower() == "true" if isinstance(DoFormant, str) else DoFormant
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "w+", "formanting", DoFormant, Quefrency, Timbre)

config = Config()
i18n = I18nAuto()
i18n.print()

# GPU detection
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() and ngpu > 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in [
            "10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", 
            "A50", "500", "A60", "70", "80", "90", "M4", "T4", "TITAN"
        ]):
            if_gpu_ok = True
            gpu_infos.append(f"{i}\t{gpu_name}")
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))

if if_gpu_ok and gpu_infos:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
    
gpus = "-".join([i[0] for i in gpu_infos]) if gpu_infos else ""

def load_hubert():
    """Load Hubert model"""
    global hubert_model
    hubert_model = load_model("hubert_base.pt")

# Path constants
WEIGHT_ROOT = "weights"
UVR5_ROOT = "uvr5_weights"
INDEX_ROOT = "./logs/"
AUDIO_ROOT = "audios"

def get_model_names():
    """Get list of model names"""
    models = []
    if os.path.exists(WEIGHT_ROOT):
        models = [f for f in os.listdir(WEIGHT_ROOT) if f.endswith(".pth")]
    return sorted(models)

def get_index_paths():
    """Get all index paths"""
    indexes = []
    if os.path.exists(INDEX_ROOT):
        for root, _, files in os.walk(INDEX_ROOT):
            for file in files:
                if file.endswith(".index") and "trained" not in file:
                    indexes.append(os.path.join(root, file).replace("\\", "/"))
    return indexes

def get_audio_paths():
    """Get all audio paths"""
    audio_files = []
    if os.path.exists(AUDIO_ROOT):
        for file in os.listdir(AUDIO_ROOT):
            audio_files.append(os.path.join(AUDIO_ROOT, file).replace("\\", "/"))
    return sorted(audio_files)

def get_uvr5_names():
    """Get UVR5 model names"""
    names = []
    if os.path.exists(UVR5_ROOT):
        names = [f.replace(".pth", "") for f in os.listdir(UVR5_ROOT) 
                if f.endswith(".pth") or "onnx" in f]
    return names

def get_fshift_presets():
    """Get formant shift presets"""
    presets = []
    preset_dir = "./formantshiftcfg"
    if os.path.exists(preset_dir):
        for root, _, files in os.walk(preset_dir):
            for file in files:
                if file.endswith(".txt"):
                    presets.append(os.path.join(root, file).replace("\\", "/"))
    return presets

def get_index_for_model(model_name):
    """Get index file for a specific model"""
    if not model_name:
        return ""
    
    model_base = model_name.split(".")[0].split("_")[0]
    logs_path = os.path.join("logs", model_base)
    
    if os.path.exists(logs_path):
        for file in os.listdir(logs_path):
            if file.endswith(".index"):
                return os.path.join(logs_path, file).replace("\\", "/")
    return ""

def vc_single(sid, input_audio_path0, input_audio_path1, f0_up_key, f0_file,
              f0_method, file_index, file_index2, index_rate, filter_radius,
              resample_sr, rms_mix_rate, protect, crepe_hop_length):
    """Single file voice conversion"""
    global tgt_sr, net_g, vc, hubert_model, version
    
    if not input_audio_path0 and not input_audio_path1:
        return "You need to upload an audio", None
    
    f0_up_key = int(f0_up_key)
    
    try:
        # Load audio
        audio_path = input_audio_path0 if input_audio_path0 else input_audio_path1
        audio = load_audio(audio_path, 16000, DoFormant, Quefrency, Timbre)
        
        # Normalize audio
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
            
        times = [0, 0, 0]
        
        if hubert_model is None:
            load_hubert()
            
        if_f0 = cpt.get("f0", 1)
        
        # Clean up index path
        file_index = file_index.strip(' "\n') if file_index else file_index2
        file_index = file_index.replace("trained", "added") if file_index else file_index2
        
        audio_opt = vc.pipeline(
            hubert_model, net_g, sid, audio, input_audio_path1, times,
            f0_up_key, f0_method, file_index, index_rate, if_f0,
            filter_radius, tgt_sr, resample_sr, rms_mix_rate, version,
            protect, crepe_hop_length, f0_file=f0_file
        )
        
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
            
        index_info = f"Using index:{file_index}." if os.path.exists(file_index) else "Index not used."
        
        return (f"Success.\n {index_info}\nTime:\n npy:{times[0]}s, f0:{times[1]}s, infer:{times[2]}s", 
                (tgt_sr, audio_opt))
                
    except Exception:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

def vc_multi(sid, dir_path, opt_root, paths, f0_up_key, f0_method,
             file_index, file_index2, index_rate, filter_radius,
             resample_sr, rms_mix_rate, protect, format1, crepe_hop_length):
    """Batch voice conversion"""
    try:
        dir_path = dir_path.strip(' "\n')
        opt_root = opt_root.strip(' "\n')
        os.makedirs(opt_root, exist_ok=True)
        
        # Get file paths
        if dir_path and os.path.exists(dir_path):
            paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        else:
            paths = [p.name for p in paths] if paths else []
            
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid, path, None, f0_up_key, None, f0_method,
                file_index, file_index2, index_rate, filter_radius,
                resample_sr, rms_mix_rate, protect, crepe_hop_length
            )
            
            if "Success" in info and opt and opt[1] is not None:
                try:
                    tgt_sr, audio_opt = opt
                    base_name = os.path.basename(path)
                    
                    if format1 in ["wav", "flac", "mp3", "ogg", "aac"]:
                        out_path = os.path.join(opt_root, f"{base_name}.{format1}")
                        sf.write(out_path, audio_opt, tgt_sr)
                    else:
                        wav_path = os.path.join(opt_root, f"{base_name}.wav")
                        sf.write(wav_path, audio_opt, tgt_sr)
                        if os.path.exists(wav_path):
                            out_path = wav_path[:-4] + f".{format1}"
                            os.system(f"ffmpeg -i {wav_path} -vn {out_path} -q:a 2 -y")
                except Exception:
                    info += traceback.format_exc()
                    
            infos.append(f"{os.path.basename(path)}->{info}")
            yield "\n".join(infos)
            
    except Exception:
        yield traceback.format_exc()

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    """UVR5 audio separation"""
    infos = []
    try:
        inp_root = inp_root.strip(' "\n')
        save_root_vocal = save_root_vocal.strip(' "\n')
        save_root_ins = save_root_ins.strip(' "\n')
        
        # Initialize model
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = _audio_pre_ if "DeEcho" not in model_name else _audio_pre_new
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(UVR5_ROOT, model_name + ".pth"),
                device=config.device,
                is_half=config.is_half,
            )
        
        # Get file paths
        if inp_root and os.path.exists(inp_root):
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [p.name for p in paths] if paths else []
        
        for path in paths:
            inp_path = os.path.join(inp_root, path) if inp_root else path
            need_reformat = 1
            done = 0
            
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                    need_reformat = 0
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                    done = 1
            except:
                need_reformat = 1
                
            if need_reformat == 1:
                tmp_path = os.path.join(TEMP_DIR, f"{os.path.basename(inp_path)}.reformatted.wav")
                os.system(f"ffmpeg -i {inp_path} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} -y")
                inp_path = tmp_path
                
            try:
                if done == 0:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0)
                infos.append(f"{os.path.basename(inp_path)}->Success")
                yield "\n".join(infos)
            except:
                infos.append(f"{os.path.basename(inp_path)}->{traceback.format_exc()}")
                yield "\n".join(infos)
                
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        # Cleanup
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            pass
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_vc(sid, to_return_protect0, to_return_protect1):
    """Load voice conversion model"""
    global n_spk, tgt_sr, net_g, vc, cpt, version, hubert_model
    
    if not sid:
        # Clean up models
        if hubert_model is not None:
            del net_g, n_spk, vc, hubert_model, tgt_sr
            hubert_model = net_g = n_spk = vc = tgt_sr = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if cpt:
                if_f0 = cpt.get("f0", 1)
                version = cpt.get("version", "v1")
                
                if version == "v1":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                else:
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                        
                del net_g, cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cpt = None
                
        return ({"visible": False, "__type__": "update"}, 
                {"visible": False, "__type__": "update"}, 
                {"visible": False, "__type__": "update"})
    
    # Load new model
    person = os.path.join(WEIGHT_ROOT, sid)
    print(f"loading {person}")
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    
    if if_f0 == 0:
        to_return_protect0 = to_return_protect1 = {"visible": False, "value": 0.5, "__type__": "update"}
    else:
        to_return_protect0 = {"visible": True, "value": to_return_protect0, "__type__": "update"}
        to_return_protect1 = {"visible": True, "value": to_return_protect1, "__type__": "update"}
    
    version = cpt.get("version", "v1")
    
    # Initialize model
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    else:
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
        
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    
    return ({"visible": True, "maximum": n_spk, "__type__": "update"}, 
            to_return_protect0, 
            to_return_protect1)

def change_choices():
    """Refresh dropdown choices"""
    return (
        {"choices": get_model_names(), "__type__": "update"},
        {"choices": get_index_paths(), "__type__": "update"},
        {"choices": get_audio_paths(), "__type__": "update"},
    )

def clean():
    """Clear input"""
    return {"value": "", "__type__": "update"}

def formant_enabled(cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button):
    """Handle formant enable/disable"""
    global DoFormant, Quefrency, Timbre
    
    DoFormant = cbox
    Quefrency = qfrency if cbox else 1.0
    Timbre = tmbre if cbox else 1.0
    
    CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "w+", "formanting", DoFormant, Quefrency, Timbre)
    
    return (
        {"value": cbox, "__type__": "update"},
        {"visible": cbox, "__type__": "update"},
        {"visible": cbox, "__type__": "update"},
        {"visible": cbox, "__type__": "update"},
        {"visible": cbox, "__type__": "update"},
        {"visible": cbox, "__type__": "update"},
    )

def formant_apply(qfrency, tmbre):
    """Apply formant settings"""
    global Quefrency, Timbre, DoFormant
    
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    
    CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "w+", "formanting", DoFormant, qfrency, tmbre)
    
    return ({"value": qfrency, "__type__": "update"}, 
            {"value": tmbre, "__type__": "update"})

def update_fshift_presets(preset, qfrency, tmbre):
    """Update formant presets"""
    if preset:
        try:
            with open(str(preset), "r") as p:
                content = p.readlines()
                qfrency = float(content[0].strip())
                tmbre = float(content[1].strip())
                formant_apply(qfrency, tmbre)
        except:
            pass
            
    return ({"choices": get_fshift_presets(), "__type__": "update"}, 
            {"value": qfrency, "__type__": "update"}, 
            {"value": tmbre, "__type__": "update"})

def if_done(done, p):
    """Check if process is done"""
    while True:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    """Check if multiple processes are done"""
    while True:
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    """Preprocess dataset for training"""
    sr = sr_dict[sr]
    log_dir = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "preprocess.log")
    open(log_path, "w").close()
    
    cmd = f"{config.python_cmd} trainset_preprocess_pipeline_print.py {trainset_dir} {sr} {n_p} {log_dir} {config.noparallel}"
    print(cmd)
    
    p = Popen(cmd, shell=True)
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()
    
    while not done[0]:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                yield f.read()
        sleep(1)
        
    with open(log_path, "r") as f:
        yield f.read()

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    """Extract F0 and features"""
    gpus = gpus.split("-") if gpus else []
    log_dir = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "extract_f0_feature.log")
    open(log_path, "w").close()
    
    ps = []
    
    if if_f0:
        cmd = f"{config.python_cmd} extract_f0_print.py {log_dir} {n_p} {f0method} {echl}"
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    
    # Extract features for each GPU
    for idx, n_g in enumerate(gpus):
        cmd = f"{config.python_cmd} extract_feature_print.py {config.device} {len(gpus)} {idx} {n_g} {log_dir} {version19}"
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    
    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps)).start()
    
    while not done[0]:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                yield f.read()
        sleep(1)
        
    with open(log_path, "r") as f:
        yield f.read()

def change_sr2(sr2, if_f0_3, version19):
    """Update pretrained paths based on sample rate"""
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    
    g_path = f"pretrained{path_str}/{f0_str}G{sr2}.pth"
    d_path = f"pretrained{path_str}/{f0_str}D{sr2}.pth"
    
    return (g_path if os.path.exists(g_path) else "", 
            d_path if os.path.exists(d_path) else "")

def change_version19(sr2, if_f0_3, version19):
    """Handle version change"""
    path_str = "" if version19 == "v1" else "_v2"
    
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
        
    sr_choices = ["40k", "48k", "32k"] if version19 == "v2" else ["40k", "48k"]
    
    f0_str = "f0" if if_f0_3 else ""
    g_path = f"pretrained{path_str}/{f0_str}G{sr2}.pth"
    d_path = f"pretrained{path_str}/{f0_str}D{sr2}.pth"
    
    return (
        g_path if os.path.exists(g_path) else "",
        d_path if os.path.exists(d_path) else "",
        {"choices": sr_choices, "value": sr2, "__type__": "update"}
    )

def change_f0(if_f0_3, sr2, version19, step2b, gpus6, gpu_info9, 
              extraction_crepe_hop_length, but2, info2):
    """Handle F0 method change"""
    path_str = "" if version19 == "v1" else "_v2"
    
    if if_f0_3:
        g_path = f"assets/pretrained{path_str}/f0G{sr2}.pth"
        d_path = f"assets/pretrained{path_str}/f0D{sr2}.pth"
        return (
            {"visible": True, "__type__": "update"},
            g_path if os.path.exists(g_path) else "",
            d_path if os.path.exists(d_path) else "",
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )
    else:
        g_path = f"assets/pretrained{path_str}/G{sr2}.pth"
        d_path = f"assets/pretrained{path_str}/D{sr2}.pth"
        return (
            {"visible": False, "__type__": "update"},
            g_path if os.path.exists(g_path) else "",
            d_path if os.path.exists(d_path) else "",
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )

def set_log_interval(exp_dir, batch_size):
    """Calculate log interval based on dataset size"""
    folder_path = os.path.join(exp_dir, "1_16k_wavs")
    
    if os.path.exists(folder_path):
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        if wav_files:
            log_interval = math.ceil(len(wav_files) / batch_size)
            return log_interval + 1 if log_interval > 1 else 1
            
    return 1

def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11,
                batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
                gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    """Start training"""
    CSVutil(os.path.join(CSV_DIR, "stop.csv"), "w+", "formanting", False)
    
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    
    log_interval = set_log_interval(exp_dir, batch_size12)
    
    # Generate filelist
    if if_f0_3:
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
        names = (set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(feature_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(f0_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(f0nsf_dir)]))
    else:
        names = (set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(feature_dir)]))
    
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}/{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}/{name}.npy|"
                f"{f0_dir.replace('\\', '\\\\')}/{name}.wav.npy|"
                f"{f0nsf_dir.replace('\\', '\\\\')}/{name}.wav.npy|{spk_id5}"
            )
        else:
            opt.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}/{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}/{name}.npy|{spk_id5}"
            )
    
    fea_dim = 256 if version19 == "v1" else 768
    
    # Add mute samples
    for _ in range(2):
        if if_f0_3:
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|"
                f"{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|"
                f"{now_dir}/logs/mute/2a_f0/mute.wav.npy|"
                f"{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}"
            )
        else:
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|"
                f"{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}"
            )
    
    shuffle(opt)
    
    with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    
    print("write filelist done")
    
    # Build training command
    cmd_parts = [
        config.python_cmd,
        "train_nsf_sim_cache_sid_load_pretrain.py",
        f"-e {exp_dir1}",
        f"-sr {sr2}",
        f"-f0 {1 if if_f0_3 else 0}",
        f"-bs {batch_size12}",
        f"-g {gpus16}" if gpus16 else "",
        f"-te {total_epoch11}",
        f"-se {save_epoch10}",
        f"-pg {pretrained_G14}" if pretrained_G14 else "",
        f"-pd {pretrained_D15}" if pretrained_D15 else "",
        f"-l {1 if if_save_latest13 else 0}",
        f"-c {1 if if_cache_gpu17 else 0}",
        f"-sw {1 if if_save_every_weights18 else 0}",
        f"-v {version19}",
        f"-li {log_interval}",
    ]
    
    cmd = " ".join([p for p in cmd_parts if p])
    print(cmd)
    
    global p, PID
    p = Popen(cmd, shell=True, cwd=now_dir)
    PID = p.pid
    
    p.wait()
    
    return ("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log",
            {"visible": False, "__type__": "update"},
            {"visible": True, "__type__": "update"})

def train_index(exp_dir1, version19):
    """Train feature index"""
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    
    feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    
    if not os.path.exists(feature_dir):
        yield "请先进行特征提取!"
        return
        
    files = os.listdir(feature_dir)
    if not files:
        yield "请先进行特征提取！"
        return
    
    infos = []
    npys = []
    
    for name in sorted(files):
        phone = np.load(os.path.join(feature_dir, name))
        npys.append(phone)
        
    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)
    
    # K-means for large datasets
    if big_npy.shape[0] > 2e5:
        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * config.n_cpu,
                compute_labels=False,
                init="random",
            )
            big_npy = kmeans.fit(big_npy).cluster_centers_
        except:
            info = traceback.format_exc()
            print(info)
            infos.append(info)
            yield "\n".join(infos)
    
    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
    
    dim = 256 if version19 == "v1" else 768
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    
    # Build index
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    
    infos.append("training")
    yield "\n".join(infos)
    
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    
    trained_path = os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    faiss.write_index(index, trained_path)
    
    infos.append("adding")
    yield "\n".join(infos)
    
    # Add vectors in batches
    batch_size = 8192
    for i in range(0, big_npy.shape[0], batch_size):
        index.add(big_npy[i:i + batch_size])
        
    added_path = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    faiss.write_index(index, added_path)
    
    infos.append(f"Successful Index Construction，added_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    yield "\n".join(infos)

def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8,
              save_epoch10, total_epoch11, batch_size12, if_save_latest13,
              pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17,
              if_save_every_weights18, version19, echl):
    """One-click training pipeline"""
    infos = []
    
    def add_info(msg):
        infos.append(msg)
        return "\n".join(infos)
    
    model_log_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(model_log_dir, exist_ok=True)
    
    # Step 1: Preprocess
    yield add_info(i18n("step1:正在处理数据"))
    preprocess_log = os.path.join(model_log_dir, "preprocess.log")
    open(preprocess_log, "w").close()
    
    cmd = f"{config.python_cmd}rvc/train/trainset_preprocess_pipeline_print.py {trainset_dir4} {sr_dict[sr2]} {np7} {model_log_dir} {config.noparallel}"
    yield add_info(cmd)
    
    p = Popen(cmd, shell=True)
    p.wait()
    
    with open(preprocess_log, "r") as f:
        print(f.read())
    
    # Step 2a: Extract pitch
    extract_log = os.path.join(model_log_dir, "extract_f0_feature.log")
    open(extract_log, "w").close()
    
    if if_f0_3:
        yield add_info("step2a:正在提取音高")
        cmd = f"{config.python_cmd}rvc/tools/extract/extract_f0_print.py {model_log_dir} {np7} {f0method8} {echl}"
        yield add_info(cmd)
        
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        
        with open(extract_log, "r") as f:
            print(f.read())
    else:
        yield add_info(i18n("step2a:无需提取音高"))
    
    # Step 2b: Extract features
    yield add_info(i18n("step2b:正在提取特征"))
    
    gpus = gpus16.split("-") if gpus16 else []
    ps = []
    
    for idx, n_g in enumerate(gpus):
        cmd = f"{config.python_cmd}rvc/tools/extract/extract_feature_print.py {config.device} {len(gpus)} {idx} {n_g} {model_log_dir} {version19}"
        yield add_info(cmd)
        
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    
    for p in ps:
        p.wait()
    
    with open(extract_log, "r") as f:
        print(f.read())
    
    # Step 3a: Train model
    yield add_info(i18n("step3a:正在训练模型"))
    
    # Generate filelist
    gt_wavs_dir = os.path.join(model_log_dir, "0_gt_wavs")
    feature_dir = os.path.join(model_log_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    
    if if_f0_3:
        f0_dir = os.path.join(model_log_dir, "2a_f0")
        f0nsf_dir = os.path.join(model_log_dir, "2b-f0nsf")
        names = (set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(feature_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(f0_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(f0nsf_dir)]))
    else:
        names = (set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & 
                set([name.split(".")[0] for name in os.listdir(feature_dir)]))
    
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}/{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}/{name}.npy|"
                f"{f0_dir.replace('\\', '\\\\')}/{name}.wav.npy|"
                f"{f0nsf_dir.replace('\\', '\\\\')}/{name}.wav.npy|{spk_id5}"
            )
        else:
            opt.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}/{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}/{name}.npy|{spk_id5}"
            )
    
    fea_dim = 256 if version19 == "v1" else 768
    
    for _ in range(2):
        if if_f0_3:
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|"
                f"{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|"
                f"{now_dir}/logs/mute/2a_f0/mute.wav.npy|"
                f"{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id5}"
            )
        else:
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|"
                f"{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{spk_id5}"
            )
    
    shuffle(opt)
    
    with open(os.path.join(model_log_dir, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    
    yield add_info("write filelist done")
    
    # Training command
    cmd_parts = [
        config.python_cmd + "rvc/train/train_nsf_sim_cache_sid_load_pretrain.py",
        f"-e {exp_dir1}",
        f"-sr {sr2}",
        f"-f0 {1 if if_f0_3 else 0}",
        f"-bs {batch_size12}",
        f"-g {gpus16}" if gpus16 else "",
        f"-te {total_epoch11}",
        f"-se {save_epoch10}",
        f"-pg {pretrained_G14}" if pretrained_G14 else "",
        f"-pd {pretrained_D15}" if pretrained_D15 else "",
        f"-l {1 if if_save_latest13 else 0}",
        f"-c {1 if if_cache_gpu17 else 0}",
        f"-sw {1 if if_save_every_weights18 else 0}",
        f"-v {version19}",
    ]
    
    cmd = " ".join([p for p in cmd_parts if p])
    yield add_info(cmd)
    
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    
    yield add_info(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"))
    
    # Step 3b: Train index
    feature_files = sorted(os.listdir(feature_dir))
    npys = []
    
    for name in feature_files:
        phone = np.load(os.path.join(feature_dir, name))
        npys.append(phone)
    
    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)
    
    if big_npy.shape[0] > 2e5:
        info = f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers."
        print(info)
        yield add_info(info)
        
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * config.n_cpu,
                compute_labels=False,
                init="random",
            )
            big_npy = kmeans.fit(big_npy).cluster_centers_
        except:
            info = traceback.format_exc()
            print(info)
            yield add_info(info)
    
    np.save(os.path.join(model_log_dir, "total_fea.npy"), big_npy)
    
    dim = 256 if version19 == "v1" else 768
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    
    yield add_info(f"{big_npy.shape},{n_ivf}")
    
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    yield add_info("training index")
    
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    
    trained_path = os.path.join(model_log_dir, f"trained_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    faiss.write_index(index, trained_path)
    
    yield add_info("adding index")
    
    batch_size = 8192
    for i in range(0, big_npy.shape[0], batch_size):
        index.add(big_npy[i:i + batch_size])
    
    added_path = os.path.join(model_log_dir, f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    faiss.write_index(index, added_path)
    
    yield add_info(f"成功构建索引, added_IVF{n_ivf}_Flat_nprobe_1_{exp_dir1}_{version19}.index")
    yield add_info(i18n("全流程结束！"))

def change_info_(ckpt_path):
    """Get model info from checkpoint"""
    log_path = ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
    
    if not os.path.exists(log_path):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    
    try:
        with open(log_path, "r") as f:
            first_line = f.readline().strip().split("\t")[-1]
            info = eval(first_line)
            sr = info["sample_rate"]
            f0 = info["if_f0"]
            version = "v2" if info.get("version") == "v2" else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

def export_onnx(ModelPath, ExportedPath):
    """Export model to ONNX format"""
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768
    
    # Test inputs
    test_phone = torch.rand(1, 200, vec_channels)
    test_phone_lengths = torch.tensor([200]).long()
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)
    test_pitchf = torch.rand(1, 200)
    test_ds = torch.LongTensor([0])
    test_rnd = torch.rand(1, 192, 200)
    
    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], 
        is_half=False, 
        version=cpt.get("version", "v1")
    )
    net_g.load_state_dict(cpt["weight"], strict=False)
    
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]
    
    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    
    return "Finished"

# CLI functions
def cli_split_command(com):
    """Split command line arguments"""
    exp = r'(?:(?<=\s)|^)"(.*?)"(?=\s|$)|(\S+)'
    split_array = re.findall(exp, com)
    split_array = [group[0] if group[0] else group[1] for group in split_array]
    return split_array

def execute_generator_function(genObject):
    """Execute generator function"""
    for _ in genObject:
        pass

def cli_infer(com):
    """CLI inference command"""
    com = cli_split_command(com)
    model_name = com[0]
    source_audio_path = com[1]
    output_file_name = com[2]
    feature_index_path = com[3]
    f0_file = None
    
    speaker_id = int(com[4])
    transposition = float(com[5])
    f0_method = com[6]
    crepe_hop_length = int(com[7])
    harvest_median_filter = int(com[8])
    resample = int(com[9])
    mix = float(com[10])
    feature_ratio = float(com[11])
    protection_amnt = float(com[12])
    protect1 = 0.5
    
    if com[14].lower() == "false":
        DoFormant = False
        Quefrency = 0.0
        Timbre = 0.0
        CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "w+", "formanting", DoFormant, Quefrency, Timbre)
    else:
        DoFormant = True
        Quefrency = float(com[15])
        Timbre = float(com[16])
        CSVutil(os.path.join(CSV_DIR, "formanting.csv"), "w+", "formanting", DoFormant, Quefrency, Timbre)
    
    print("Mangio-RVC-Fork Infer-CLI: Starting the inference...")
    get_vc(model_name, protection_amnt, protect1)
    print("Mangio-RVC-Fork Infer-CLI: Performing inference...")
    
    conversion_data = vc_single(
        speaker_id, source_audio_path, source_audio_path, transposition,
        f0_file, f0_method, feature_index_path, feature_index_path,
        feature_ratio, harvest_median_filter, resample, mix,
        protection_amnt, crepe_hop_length
    )
    
    if "Success." in conversion_data[0]:
        os.makedirs("audio-outputs", exist_ok=True)
        output_path = os.path.join("audio-outputs", output_file_name)
        print(f"Mangio-RVC-Fork Infer-CLI: Writing to {output_path}...")
        wavfile.write(output_path, conversion_data[1][0], conversion_data[1][1])
        print(f"Mangio-RVC-Fork Infer-CLI: Finished! Saved output to {output_path}")
    else:
        print("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ")
        print(conversion_data[0])

def cli_pre_process(com):
    """CLI preprocess command"""
    com = cli_split_command(com)
    model_name = com[0]
    trainset_directory = com[1]
    sample_rate = com[2]
    num_processes = int(com[3])
    
    print("Mangio-RVC-Fork Pre-process: Starting...")
    generator = preprocess_dataset(trainset_directory, model_name, sample_rate, num_processes)
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Pre-process: Finished")

def cli_extract_feature(com):
    """CLI extract feature command"""
    com = cli_split_command(com)
    model_name = com[0]
    gpus = com[1]
    num_processes = int(com[2])
    has_pitch_guidance = int(com[3]) == 1
    f0_method = com[4]
    crepe_hop_length = int(com[5])
    version = com[6]
    
    print("Mangio-RVC-CLI: Extract Feature Has Pitch: " + str(has_pitch_guidance))
    print("Mangio-RVC-CLI: Extract Feature Version: " + str(version))
    print("Mangio-RVC-Fork Feature Extraction: Starting...")
    
    generator = extract_f0_feature(
        gpus, num_processes, f0_method, has_pitch_guidance,
        model_name, version, crepe_hop_length
    )
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Feature Extraction: Finished")

def cli_train(com):
    """CLI train command"""
    com = cli_split_command(com)
    model_name = com[0]
    sample_rate = com[1]
    has_pitch_guidance = int(com[2]) == 1
    speaker_id = int(com[3])
    save_epoch_iteration = int(com[4])
    total_epoch = int(com[5])
    batch_size = int(com[6])
    gpu_card_slot_numbers = com[7]
    if_save_latest = int(com[8]) == 1
    if_cache_gpu = int(com[9]) == 1
    if_save_every_weight = int(com[10]) == 1
    version = com[11]
    
    pretrained_base = "pretrained/" if version == "v1" else "pretrained_v2/"
    g_pretrained_path = f"{pretrained_base}f0G{sample_rate}.pth"
    d_pretrained_path = f"{pretrained_base}f0D{sample_rate}.pth"
    
    print("Mangio-RVC-Fork Train-CLI: Training...")
    click_train(
        model_name, sample_rate, has_pitch_guidance, speaker_id,
        save_epoch_iteration, total_epoch, batch_size, if_save_latest,
        g_pretrained_path, d_pretrained_path, gpu_card_slot_numbers,
        if_cache_gpu, if_save_every_weight, version
    )

def cli_train_feature(com):
    """CLI train feature command"""
    com = cli_split_command(com)
    model_name = com[0]
    version = com[1]
    print("Mangio-RVC-Fork Train Feature Index-CLI: Training... Please wait")
    generator = train_index(model_name, version)
    execute_generator_function(generator)
    print("Mangio-RVC-Fork Train Feature Index-CLI: Done!")

def cli_extract_model(com):
    """CLI extract model command"""
    com = cli_split_command(com)
    model_path = com[0]
    save_name = com[1]
    sample_rate = com[2]
    has_pitch_guidance = com[3]
    info = com[4]
    version = com[5]
    
    result = extract_small_model(
        model_path, save_name, sample_rate, has_pitch_guidance, info, version
    )
    
    if result == "Success.":
        print("Mangio-RVC-Fork Extract Small Model: Success!")
    else:
        print(str(result))
        print("Mangio-RVC-Fork Extract Small Model: Failed!")

def preset_apply(preset, qfer, tmbr):
    """Apply formant preset"""
    if preset:
        try:
            with open(str(preset), "r") as p:
                content = p.readlines()
                qfer = float(content[0].strip())
                tmbr = float(content[1].strip())
                formant_apply(qfer, tmbr)
        except:
            pass
            
    return {"value": qfer, "__type__": "update"}, {"value": tmbr, "__type__": "update"}

def match_index(sid0):
    """Match index file to model"""
    if not sid0:
        return "", ""
    
    folder = sid0.split(".")[0].split("_")[0]
    parent_dir = os.path.join("logs", folder)
    
    if os.path.exists(parent_dir):
        for filename in os.listdir(parent_dir):
            if filename.endswith(".index"):
                index_path = os.path.join(parent_dir, filename).replace("\\", "/")
                return index_path, index_path
    
    return "", ""

def stoptraining(mim):
    """Stop training process"""
    global p, PID
    
    if int(mim) == 1:
        CSVutil(os.path.join(CSV_DIR, "stop.csv"), "w+", "stop", "True")
        try:
            if PID:
                os.kill(PID, signal.SIGTERM)
        except Exception as e:
            print(f"Couldn't stop due to {e}")
    
    return {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}

def whethercrepeornah(radio):
    """Check if crepe method is selected"""
    is_crepe = radio in ["mangio-crepe", "mangio-crepe-tiny"]
    return {"visible": is_crepe, "__type__": "update"}

def stepdisplay(if_save_every_weights):
    """Display step based on setting"""
    return {"visible": if_save_every_weights, "__type__": "update"}

# CLI Navigation
cli_current_page = "HOME"

def print_page_details():
    """Print current page details"""
    if cli_current_page == "HOME":
        print(
            "\n    go home            : Takes you back to home with a navigation list."
            "\n    go infer           : Takes you to inference command execution."
            "\n    go pre-process     : Takes you to training step.1) pre-process command execution."
            "\n    go extract-feature : Takes you to training step.2) extract-feature command execution."
            "\n    go train           : Takes you to training step.3) being or continue training command execution."
            "\n    go train-feature   : Takes you to the train feature index command execution."
            "\n    go extract-model   : Takes you to the extract small model command execution."
        )
    elif cli_current_page == "INFER":
        print(
            "\n    arg 1) model name with .pth in ./weights: mi-test.pth"
            "\n    arg 2) source audio path: myFolder\\MySource.wav"
            "\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav"
            "\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index"
            "\n    arg 5) speaker id: 0"
            "\n    arg 6) transposition: 0"
            "\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)"
            "\n    arg 8) crepe hop length: 160"
            "\n    arg 9) harvest median filter radius: 3 (0-7)"
            "\n    arg 10) post resample rate: 0"
            "\n    arg 11) mix volume envelope: 1"
            "\n    arg 12) feature index ratio: 0.78 (0-1)"
            "\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)"
            "\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)"
            "\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)"
            "\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n"
            "\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2"
        )
    elif cli_current_page == "PRE-PROCESS":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Number of CPU threads to use: 8 \n"
            "\nExample: mi-test mydataset 40k 24"
        )
    elif cli_current_page == "EXTRACT-FEATURE":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
            "\n    arg 3) Number of CPU threads to use: 8"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            "\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)"
            "\n    arg 6) Crepe hop length: 128"
            "\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n"
            "\nExample: mi-test 0 24 1 harvest 128 v2"
        )
    elif cli_current_page == "TRAIN":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            "\n    arg 4) speaker id: 0"
            "\n    arg 5) Save epoch iteration: 50"
            "\n    arg 6) Total epochs: 10000"
            "\n    arg 7) Batch size: 8"
            "\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)"
            "\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)"
            "\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)"
            "\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)"
            "\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n"
            "\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2"
        )
    elif cli_current_page == "TRAIN-FEATURE":
        print(
            "\n    arg 1) Model folder name in ./logs: mi-test"
            "\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n"
            "\nExample: mi-test v2"
        )
    elif cli_current_page == "EXTRACT-MODEL":
        print(
            "\n    arg 1) Model Path: logs/mi-test/G_168000.pth"
            "\n    arg 2) Model save name: MyModel"
            "\n    arg 3) Sample rate: 40k (32k, 40k, 48k)"
            "\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)"
            '\n    arg 5) Model information: "My Model"'
            "\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n"
            '\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2'
        )

def change_page(page):
    """Change current CLI page"""
    global cli_current_page
    cli_current_page = page
    return 0

def execute_command(com):
    """Execute CLI command"""
    if com == "go home":
        return change_page("HOME")
    elif com == "go infer":
        return change_page("INFER")
    elif com == "go pre-process":
        return change_page("PRE-PROCESS")
    elif com == "go extract-feature":
        return change_page("EXTRACT-FEATURE")
    elif com == "go train":
        return change_page("TRAIN")
    elif com == "go train-feature":
        return change_page("TRAIN-FEATURE")
    elif com == "go extract-model":
        return change_page("EXTRACT-MODEL")
    else:
        if com[:3] == "go ":
            print(f"page '{com[3:]}' does not exist!")
            return 0
    
    if cli_current_page == "INFER":
        cli_infer(com)
    elif cli_current_page == "PRE-PROCESS":
        cli_pre_process(com)
    elif cli_current_page == "EXTRACT-FEATURE":
        cli_extract_feature(com)
    elif cli_current_page == "TRAIN":
        cli_train(com)
    elif cli_current_page == "TRAIN-FEATURE":
        cli_train_feature(com)
    elif cli_current_page == "EXTRACT-MODEL":
        cli_extract_model(com)

def cli_navigation_loop():
    """CLI navigation loop"""
    while True:
        print(f"\nYou are currently in '{cli_current_page}':")
        print_page_details()
        command = input(f"{cli_current_page}: ")
        try:
            execute_command(command)
        except:
            print(traceback.format_exc())

if config.is_cli:
    print("\n\nMangio RVC Fork v2 CLI App!\n")
    print("Welcome to the CLI version of RVC.")
    cli_navigation_loop()

# Gradio WebUI
with gr.Blocks(theme=gr.themes.Base(), title="MANGIO RVC WEB 💻") as app:
    gr.HTML("<h1> The Mangio RVC Fork 💻 </h1>")
    gr.Markdown(value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>使用需遵守的协议-LICENSE.txt</b>."))
    
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=get_model_names(), value="")
                refresh_button = gr.Button(i18n("Refresh voice list, index path and audio files"), variant="primary")
                clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label=i18n("请选择说话人id"), value=0, visible=False, interactive=True)
                
                clean_button.click(fn=clean, inputs=[], outputs=[sid0])
            
            with gr.Group():
                gr.Markdown(value=i18n("男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域."))
                
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0)
                        input_audio0 = gr.Textbox(
                            label=i18n("Add audio's name to the path to the audio file to be processed (default is the correct format example) Remove the path to use an audio from the dropdown list:"),
                            value=os.path.join(os.getcwd(), "audios", "audio.wav"),
                        )
                        input_audio1 = gr.Dropdown(
                            label=i18n("Auto detect audio path and select from the dropdown:"),
                            choices=get_audio_paths(),
                            value="",
                            interactive=True,
                        )
                        input_audio1.change(fn=lambda: "", inputs=[], outputs=[input_audio0])
                        
                        f0method0 = gr.Radio(
                            label=i18n("选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"),
                            choices=["pm", "harvest", "dio", "crepe", "crepe-tiny", 
                                    "mangio-crepe", "mangio-crepe-tiny", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        
                        crepe_hop_length = gr.Slider(
                            minimum=1, maximum=512, step=1,
                            label=i18n("crepe_hop_length"), value=120, interactive=True, visible=False
                        )
                        f0method0.change(fn=whethercrepeornah, inputs=[f0method0], outputs=[crepe_hop_length])
                        
                        filter_radius0 = gr.Slider(
                            minimum=0, maximum=7, step=1,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3, interactive=True
                        )
                    
                    with gr.Column():
                        file_index1 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="", interactive=True
                        )
                        file_index2 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_index_paths(), value=get_index_for_model(""), interactive=True, allow_custom_value=True
                        )
                        
                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index2, input_audio1],
                        )
                        
                        index_rate1 = gr.Slider(
                            minimum=0, maximum=1,
                            label=i18n("检索特征占比"), value=0.75, interactive=True
                        )
                    
                    with gr.Column():
                        resample_sr0 = gr.Slider(
                            minimum=0, maximum=48000, step=1,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"), value=0, interactive=True
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0, maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"), value=0.25, interactive=True
                        )
                        protect0 = gr.Slider(
                            minimum=0, maximum=0.5, step=0.01,
                            label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"), 
                            value=0.33, interactive=True
                        )
                        
                        formanting = gr.Checkbox(
                            value=DoFormant,
                            label="[EXPERIMENTAL] Formant shift inference audio",
                            info="Used for male to female and vice-versa conversions",
                            interactive=True, visible=True
                        )
                        
                        formant_preset = gr.Dropdown(
                            value="", choices=get_fshift_presets(),
                            label="browse presets for formanting", visible=DoFormant
                        )
                        
                        formant_refresh_button = gr.Button(
                            value="\U0001f504", visible=DoFormant, variant="primary"
                        )
                        
                        qfrency = gr.Slider(
                            value=Quefrency, minimum=0.0, maximum=16.0, step=0.1,
                            label="Quefrency for formant shifting", info="Default value is 1.0",
                            visible=DoFormant, interactive=True
                        )
                        
                        tmbre = gr.Slider(
                            value=Timbre, minimum=0.0, maximum=16.0, step=0.1,
                            label="Timbre for formant shifting", info="Default value is 1.0",
                            visible=DoFormant, interactive=True
                        )
                        
                        formant_preset.change(
                            fn=preset_apply,
                            inputs=[formant_preset, qfrency, tmbre],
                            outputs=[qfrency, tmbre],
                        )
                        
                        frmntbut = gr.Button("Apply", variant="primary", visible=DoFormant)
                        
                        formanting.change(
                            fn=formant_enabled,
                            inputs=[formanting, qfrency, tmbre, frmntbut, formant_preset, formant_refresh_button],
                            outputs=[formanting, qfrency, tmbre, frmntbut, formant_preset, formant_refresh_button],
                        )
                        
                        frmntbut.click(
                            fn=formant_apply,
                            inputs=[qfrency, tmbre],
                            outputs=[qfrency, tmbre],
                        )
                        
                        formant_refresh_button.click(
                            fn=update_fshift_presets,
                            inputs=[formant_preset, qfrency, tmbre],
                            outputs=[formant_preset, qfrency, tmbre],
                        )
                    
                    f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"))
                    but0 = gr.Button(i18n("转换"), variant="primary")
                    
                    with gr.Row():
                        vc_output1 = gr.Textbox(label=i18n("输出信息"))
                        vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
                    
                    but0.click(
                        vc_single,
                        [spk_item, input_audio0, input_audio1, vc_transform0, f0_file,
                         f0method0, file_index1, file_index2, index_rate1, filter_radius0,
                         resample_sr0, rms_mix_rate0, protect0, crepe_hop_length],
                        [vc_output1, vc_output2],
                    )
            
            # Batch conversion section
            with gr.Group():
                gr.Markdown(value=i18n("批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频."))
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0)
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                        f0method1 = gr.Radio(
                            label=i18n("选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"),
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe", interactive=True
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0, maximum=7, step=1,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3, interactive=True
                        )
                    
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="", interactive=True
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=get_index_paths(), value=get_index_for_model(""), interactive=True
                        )
                        sid0.select(fn=match_index, inputs=[sid0], outputs=[file_index2, file_index4])
                        refresh_button.click(fn=lambda: change_choices()[1], inputs=[], outputs=file_index4)
                        
                        index_rate2 = gr.Slider(
                            minimum=0, maximum=1,
                            label=i18n("检索特征占比"), value=1, interactive=True
                        )
                    
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0, maximum=48000, step=1,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"), value=0, interactive=True
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0, maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"), value=1, interactive=True
                        )
                        protect1 = gr.Slider(
                            minimum=0, maximum=0.5, step=0.01,
                            label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"), 
                            value=0.33, interactive=True
                        )
                    
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value=os.path.join(os.getcwd(), "audios"),
                        )
                        inputs = gr.File(file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"))
                    
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac", interactive=True
                        )
                        but1 = gr.Button(i18n("转换"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("输出信息"))
                    
                    but1.click(
                        vc_multi,
                        [spk_item, dir_input, opt_input, inputs, vc_transform1, f0method1,
                         file_index3, file_index4, index_rate2, filter_radius1,
                         resample_sr1, rms_mix_rate1, protect1, format1, crepe_hop_length],
                        [vc_output3],
                    )
            
            sid0.change(
                fn=get_vc,
                inputs=[sid0, protect0, protect1],
                outputs=[spk_item, protect0, protect1],
            )
        
        with gr.TabItem(i18n("伴奏人声分离&去混响&去回声")):
            with gr.Group():
                gr.Markdown(value=i18n("人声伴奏分离批量处理， 使用UVR5模型。 <br>"
                    "合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>"
                    "模型分为三类： <br>"
                    "1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>"
                    "2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> "
                    "3、去混响、去延迟模型（by FoxJoy）：<br>"
                    "  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>"
                    "&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>"
                    "去混响/去延迟，附：<br>"
                    "1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>"
                    "2、MDX-Net-Dereverb模型挺慢的；<br>"
                    "3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive."))
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径"),
                            value=os.path.join(os.getcwd(), "audios"),
                        )
                        wav_inputs = gr.File(file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"))
                    
                    with gr.Column():
                        model_choose = gr.Dropdown(label=i18n("模型"), choices=get_uvr5_names())
                        agg = gr.Slider(minimum=0, maximum=20, step=1, label="人声提取激进程度", value=10, interactive=True, visible=False)
                        opt_vocal_root = gr.Textbox(label=i18n("指定输出主人声文件夹"), value="opt")
                        opt_ins_root = gr.Textbox(label=i18n("指定输出非主人声文件夹"), value="opt")
                        format0 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac", interactive=True
                        )
                    
                    but2 = gr.Button(i18n("转换"), variant="primary")
                    vc_output4 = gr.Textbox(label=i18n("输出信息"))
                    
                    but2.click(
                        uvr,
                        [model_choose, dir_wav_input, opt_vocal_root, wav_inputs,
                         opt_ins_root, agg, format0],
                        [vc_output4],
                    )
        
        with gr.TabItem(i18n("训练")):
            gr.Markdown(value=i18n("step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件."))
            with gr.Row():
                exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="mi-test")
                sr2 = gr.Radio(label=i18n("目标采样率"), choices=["40k", "48k"], value="40k", interactive=True)
                if_f0_3 = gr.Checkbox(label="Whether the model has pitch guidance.", value=True, interactive=True)
                version19 = gr.Radio(label=i18n("版本"), choices=["v1", "v2"], value="v1", interactive=True, visible=True)
                np7 = gr.Slider(
                    minimum=0, maximum=config.n_cpu, step=1,
                    label=i18n("提取音高和处理数据使用的CPU进程数"), 
                    value=int(np.ceil(config.n_cpu / 1.5)), interactive=True
                )
            
            with gr.Group():
                gr.Markdown(value=i18n("step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练."))
                with gr.Row():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"),
                        value=os.path.join(os.getcwd(), "datasets"),
                    )
                    spk_id5 = gr.Slider(minimum=0, maximum=4, step=1, label=i18n("请指定说话人id"), value=0, interactive=True)
                    but1 = gr.Button(i18n("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=i18n("输出信息"), value="")
                    
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                    )
            
            with gr.Group():
                step2b = gr.Markdown(value=i18n("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"))
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=gpus, interactive=True
                        )
                        gpu_info9 = gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                    
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label=i18n("选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"),
                            choices=["pm", "harvest", "dio", "crepe", "mangio-crepe", "rmvpe"],
                            value="rmvpe", interactive=True
                        )
                        
                        extraction_crepe_hop_length = gr.Slider(
                            minimum=1, maximum=512, step=1,
                            label=i18n("crepe_hop_length"), value=64, interactive=True, visible=False
                        )
                        
                        f0method8.change(
                            fn=whethercrepeornah,
                            inputs=[f0method8],
                            outputs=[extraction_crepe_hop_length],
                        )
                    
                    but2 = gr.Button(i18n("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8, interactive=False)
                    
                    but2.click(
                        extract_f0_feature,
                        [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                        [info2],
                    )
            
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(minimum=1, maximum=50, step=1, label=i18n("保存频率save_every_epoch"), value=5, interactive=True, visible=True)
                    total_epoch11 = gr.Slider(minimum=1, maximum=10000, step=1, label=i18n("总训练轮数total_epoch"), value=20, interactive=True)
                    batch_size12 = gr.Slider(minimum=1, maximum=40, step=1, label=i18n("每张显卡的batch_size"), value=default_batch_size, interactive=True)
                    if_save_latest13 = gr.Checkbox(label="Whether to save only the latest .ckpt file to save hard drive space", value=True, interactive=True)
                    if_cache_gpu17 = gr.Checkbox(label="Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement", value=False, interactive=True)
                    if_save_every_weights18 = gr.Checkbox(label="Save a small final model to the 'weights' folder at each save point", value=True, interactive=True)
                
                with gr.Row():
                    pretrained_G14 = gr.Textbox(lines=2, label=i18n("加载预训练底模G路径"), value="pretrained/f0G40k.pth", interactive=True)
                    pretrained_D15 = gr.Textbox(lines=2, label=i18n("加载预训练底模D路径"), value="pretrained/f0D40k.pth", interactive=True)
                    
                    sr2.change(change_sr2, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15])
                    
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    
                    if_f0_3.change(
                        fn=change_f0,
                        inputs=[if_f0_3, sr2, version19, step2b, gpus6, gpu_info9,
                               extraction_crepe_hop_length, but2, info2],
                        outputs=[f0method8, pretrained_G14, pretrained_D15, step2b,
                                gpus6, gpu_info9, extraction_crepe_hop_length, but2, info2],
                    )
                    
                    if_f0_3.change(
                        fn=whethercrepeornah,
                        inputs=[f0method8],
                        outputs=[extraction_crepe_hop_length],
                    )
                    
                    gpus16 = gr.Textbox(label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"), value=gpus, interactive=True)
                    
                    butstop = gr.Button("Stop Training", variant="primary", visible=False)
                    but3 = gr.Button(i18n("训练模型"), variant="primary", visible=True)
                    
                    but3.click(
                        fn=stoptraining,
                        inputs=[gr.Number(value=0, visible=False)],
                        outputs=[but3, butstop],
                    )
                    
                    butstop.click(
                        fn=stoptraining,
                        inputs=[gr.Number(value=1, visible=False)],
                        outputs=[butstop, but3],
                    )
                    
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                    
                    if_save_every_weights18.change(
                        fn=stepdisplay,
                        inputs=[if_save_every_weights18],
                        outputs=[save_epoch10],
                    )
                    
                    but3.click(
                        click_train,
                        [exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11,
                         batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
                         gpus16, if_cache_gpu17, if_save_every_weights18, version19],
                        [info3, butstop, but3],
                    )
                    
                    but4.click(train_index, [exp_dir1, version19], info3)
        
        with gr.TabItem(i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(label=i18n("A模型路径"), value="", interactive=True, placeholder="Path to your model A.")
                    ckpt_b = gr.Textbox(label=i18n("B模型路径"), value="", interactive=True, placeholder="Path to your model B.")
                    alpha_a = gr.Slider(minimum=0, maximum=1, label=i18n("A模型权重"), value=0.5, interactive=True)
                
                with gr.Row():
                    sr_ = gr.Radio(label=i18n("目标采样率"), choices=["40k", "48k"], value="40k", interactive=True)
                    if_f0_ = gr.Checkbox(label="Whether the model has pitch guidance.", value=True, interactive=True)
                    info__ = gr.Textbox(label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model information to be placed.")
                    name_to_save0 = gr.Textbox(label=i18n("保存的模型名不带后缀"), value="", placeholder="Name for saving.", max_lines=1, interactive=True)
                    version_2 = gr.Radio(label=i18n("模型版本型号"), choices=["v1", "v2"], value="v1", interactive=True)
                
                with gr.Row():
                    but6 = gr.Button(i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    
                    but6.click(
                        merge,
                        [ckpt_a, ckpt_b, alpha_a, sr_, if_f0_, info__, name_to_save0, version_2],
                        info4,
                    )
            
            with gr.Group():
                gr.Markdown(value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)"))
                with gr.Row():
                    ckpt_path0 = gr.Textbox(label=i18n("模型路径"), placeholder="Path to your Model.", value="", interactive=True)
                    info_ = gr.Textbox(label=i18n("要改的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model information to be changed.")
                    name_to_save1 = gr.Textbox(label=i18n("保存的文件名, 默认空为和源文件同名"), placeholder="Either leave empty or put in the Name of the Model to be saved.", value="", max_lines=8, interactive=True)
                
                with gr.Row():
                    but7 = gr.Button(i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    
                    but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5)
            
            with gr.Group():
                gr.Markdown(value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)"))
                with gr.Row():
                    ckpt_path1 = gr.Textbox(label=i18n("模型路径"), value="", interactive=True, placeholder="Model path here.")
                    but8 = gr.Button(i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    
                    but8.click(show_info, [ckpt_path1], info6)
            
            with gr.Group():
                gr.Markdown(value=i18n("模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"))
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        lines=3,
                        label=i18n("模型路径"),
                        value=os.path.join(os.getcwd(), "logs", "[YOUR_MODEL]", "G_23333.pth"),
                        interactive=True
                    )
                    save_name = gr.Textbox(label=i18n("保存名"), value="", interactive=True, placeholder="Your filename here.")
                    sr__ = gr.Radio(label=i18n("目标采样率"), choices=["32k", "40k", "48k"], value="40k", interactive=True)
                    if_f0__ = gr.Checkbox(label="Whether the model has pitch guidance.", value=True, interactive=True)
                    version_1 = gr.Radio(label=i18n("模型版本型号"), choices=["v1", "v2"], value="v2", interactive=True)
                    info___ = gr.Textbox(label=i18n("要置入的模型信息"), value="", max_lines=8, interactive=True, placeholder="Model info here.")
                    
                    but9 = gr.Button(i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                    
                    but9.click(
                        extract_small_model,
                        [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                        info7,
                    )
        
        with gr.TabItem(i18n("Onnx导出")):
            with gr.Row():
                ckpt_dir = gr.Textbox(label=i18n("RVC模型路径"), value="", interactive=True, placeholder="RVC model path.")
                onnx_dir = gr.Textbox(label=i18n("Onnx输出路径"), value="", interactive=True, placeholder="Onnx model output path.")
                infoOnnx = gr.Label(label="info")
                butOnnx = gr.Button(i18n("导出Onnx模型"), variant="primary")
                
                butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx)

    # Launch the app
    if config.iscolab or config.paperspace:
        app.launch(share=True)
    else:
        app.launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=False,

        )


