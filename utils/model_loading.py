from infer_d_fine.D_FINE.src.core import YAMLConfig
import os
import urllib.request
import torch
import yaml


def get_config_path(root_dir, model_name, dataset):
    # Extract model size from the last character of model_name
    model_size = model_name[-1]

    # Define the base configuration directory
    root_config = os.path.join(root_dir, "D_FINE", "configs", "dfine")

    # Handle the case where model_size is 'n' and it's not a coco dataset
    if model_size == 'n' and dataset != "coco":
        raise ValueError("Model size 'n' is only valid for the coco dataset.")

    # Determine the appropriate configuration file based on the dataset
    if dataset == "obj2coco":
        config_file = os.path.join(
            root_config, "objects365", f"dfine_hgnetv2_{model_size}_obj2coco.yml")
    elif dataset == "coco":
        config_file = os.path.join(
            root_config, f"dfine_hgnetv2_{model_size}_coco.yml")
    elif dataset == "obj365":
        config_file = os.path.join(
            root_config, "objects365", f"dfine_hgnetv2_{model_size}_obj365.yml")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return config_file


def load_custom_config(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.unsafe_load(f)


def load_model(param, device):
    # Load from costum model
    if param.model_weight_file:
        if not param.config_file:
            raise ValueError(
                "The 'config_file' is required when using a custom model file.")
        model_weights = param.model_weight_file
        cfg = load_custom_config(param.config_file)
    # Load from pre-trained
    else:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.dirname(current_dir)
        model_folder = os.path.join(root_dir, "weights")
        model_weights = os.path.join(
            model_folder, f'{param.model_name}_{param.pretrained_dataset}.pth')

        # Ensure weights directory exists
        os.makedirs(model_folder, exist_ok=True)

        # Check if the model file exists, download if not
        if not os.path.isfile(model_weights):
            url = f'https://github.com/Peterande/storage/releases/download/dfinev1.0/{param.model_name}_{param.pretrained_dataset}.pth'
            print(f"Downloading model from {url}...")
            try:
                urllib.request.urlretrieve(url, model_weights)
                print(f"Model downloaded and saved to {model_weights}")
            except Exception as e:
                print(f"Failed to download the model: {e}")
                return None, None

        # Load configuration
        config_file = get_config_path(
            root_dir, param.model_name, param.pretrained_dataset)
        cfg = YAMLConfig(config_file, resume=model_weights)

    # Disable pretrained option if HGNetv2 is in the configuration
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # Load model weights
    try:
        checkpoint = torch.load(model_weights, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None

    # Deploy model and postprocessor
    model = cfg.model.deploy().to(device)
    postprocessor = cfg.postprocessor.deploy()
    return model, postprocessor
