import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cpubone_model import get_cpubone
from ptflops import get_model_complexity_info

models = ["cpubone_nano", "cpubone_t0", "cpubone_s0"]

for name in models:
    config_path = f"configs/cls/imagenet/{name}.yaml"
    model = get_cpubone(config_path=config_path, pretrained=False)
    model.eval()

    macs, params = get_model_complexity_info(
        model, (3, 224, 224),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"{name}: MACs={macs/1e6:.1f}M  Params={params/1e6:.2f}M")
