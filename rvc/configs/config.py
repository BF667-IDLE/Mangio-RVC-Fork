import argparse
import sys
import torch
import json
import os
from multiprocessing import cpu_count

global usefp16
usefp16 = False


def use_fp32_config():
    usefp16 = False
    device_capability = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Assuming you have only one GPU (index 0).
        device_capability = torch.cuda.get_device_capability(device)[0]
        
        # Determine which config version to use based on device capability
        config_version = "v1" if device_capability < 7 else "v2"
        
        if device_capability >= 7:
            usefp16 = True
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                # Try to load from v2 first, fall back to v1 if not found
                config_path = f"configs/{config_version}/{config_file}"
                if os.path.exists(config_path):
                    with open(config_path, "r") as d:
                        data = json.load(d)
                else:
                    # Fallback to v1 if v2 doesn't exist
                    with open(f"configs/v1/{config_file}", "r") as d:
                        data = json.load(d)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = True

                # Save to the appropriate version directory
                os.makedirs(f"configs/{config_version}", exist_ok=True)
                with open(f"configs/{config_version}/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)

                print(f"Set fp16_run to true in {config_version}/{config_file}")

            with open(
                "rvc/train/trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.0", "3.7")

            with open(
                "rvc/train/trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
        else:
            for config_file in ["32k.json", "40k.json", "48k.json"]:
                # Try to load from v1 first, fall back to v2 if not found
                config_path = f"configs/{config_version}/{config_file}"
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        data = json.load(f)
                else:
                    # Fallback to v2 if v1 doesn't exist
                    with open(f"configs/v2/{config_file}", "r") as f:
                        data = json.load(f)

                if "train" in data and "fp16_run" in data["train"]:
                    data["train"]["fp16_run"] = False

                # Save to the appropriate version directory
                os.makedirs(f"configs/{config_version}", exist_ok=True)
                with open(f"configs/{config_version}/{config_file}", "w") as d:
                    json.dump(data, d, indent=4)

                print(f"Set fp16_run to false in {config_version}/{config_file}")

            with open(
                "rvc/train/trainset_preprocess_pipeline_print.py", "r", encoding="utf-8"
            ) as f:
                strr = f.read()

            strr = strr.replace("3.7", "3.0")

            with open(
                "rvc/train/trainset_preprocess_pipeline_print.py", "w", encoding="utf-8"
            ) as f:
                f.write(strr)
    else:
        print(
            "CUDA is not available. Make sure you have an NVIDIA GPU and CUDA installed."
        )
    return (usefp16, device_capability)


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        # Update to unpack 8 values
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.paperspace,
            self.is_cli,
            self.config_version,  # Add this variable to store the config version
        ) = self.arg_parse()

        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
        
        # Load the appropriate config files based on version
        self.load_config_files()

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--paperspace",
            action="store_true",
            help="Note that this argument just shares a gradio link for the web UI. Thus can be used on other non-local CLI systems.",
        )
        parser.add_argument(
            "--is_cli",
            action="store_true",
            help="Use the CLI instead of setting up a gradio UI. This flag will launch an RVC text interface where you can execute functions from infer-web.py!",
        )
        # Add command line argument to specify config version
        parser.add_argument(
            "--config_version",
            type=str,
            choices=["v1", "v2", "auto"],
            default="auto",
            help="Specify which config version to use (v1, v2, or auto-detect)"
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.paperspace,
            cmd_opts.is_cli,
            cmd_opts.config_version,  # Add config_version to return tuple
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        # Determine config version if set to auto
        config_version = getattr(self, 'config_version', 'auto')
        
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            
            # Auto-detect version based on GPU capability if set to auto
            if config_version == 'auto':
                device_capability = torch.cuda.get_device_capability(i_device)[0]
                config_version = "v2" if device_capability >= 7 else "v1"
            
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                print("Found GPU", self.gpu_name)
                use_fp32_config()
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            print("No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
            use_fp32_config()
            config_version = "v1" if config_version == 'auto' else config_version
        else:
            print("No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()
            config_version = "v1" if config_version == 'auto' else config_version

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

    def load_config_files(self):
        """Load configuration files from the appropriate version directory"""
        config_files = ["32k.json", "40k.json", "48k.json"]
        self.configs = {}
        
        # Determine which version to use (this should be set from arg_parse)
        config_version = getattr(self, 'config_version', 'auto')
        
        # If still auto, determine based on device
        if config_version == 'auto':
            if torch.cuda.is_available():
                i_device = int(self.device.split(":")[-1])
                device_capability = torch.cuda.get_device_capability(i_device)[0]
                config_version = "v2" if device_capability >= 7 else "v1"
            else:
                config_version = "v1"
        
        print(f"Loading config files from configs/{config_version}/")
        
        for config_file in config_files:
            config_path = f"configs/{config_version}/{config_file}"
            fallback_path = f"configs/v1/{config_file}" if config_version == "v2" else f"configs/v2/{config_file}"
            
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        self.configs[config_file] = json.load(f)
                    print(f"Loaded {config_file} from {config_version}")
                elif os.path.exists(fallback_path):
                    # Fallback to the other version if specified version doesn't exist
                    with open(fallback_path, "r") as f:
                        self.configs[config_file] = json.load(f)
                    print(f"Warning: {config_file} not found in {config_version}, loaded from fallback")
                else:
                    print(f"Warning: {config_file} not found in either configs/v1/ or configs/v2/")
                    self.configs[config_file] = {}
            except Exception as e:
                print(f"Error loading {config_file}: {e}")
                self.configs[config_file] = {}
        
        return self.configs


