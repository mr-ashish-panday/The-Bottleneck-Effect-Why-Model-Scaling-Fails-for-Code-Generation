from __future__ import annotations

import platform
import subprocess


def main() -> None:
    print("python_platform:", platform.platform())
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(
                    f"gpu[{idx}]: {props.name}, memory_gb={props.total_memory / (1024**3):.1f}, "
                    f"capability={props.major}.{props.minor}"
                )
    except Exception as exc:
        print("torch_probe_error:", repr(exc))

    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        print(out)
    except Exception as exc:
        print("nvidia_smi_error:", repr(exc))


if __name__ == "__main__":
    main()

