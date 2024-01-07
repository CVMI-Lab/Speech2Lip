import os 
import yaml
import logging
from torchvision import transforms

from src import face_simple
from src import data

method_dict = {
    "face_simple": face_simple
}

# General config
def load_config(path, default_path=None, abs_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    if abs_path is not None:
        path = os.path.join(abs_path, path)
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')
    if abs_path is not None:
        inherit_from = os.path.join(abs_path, inherit_from)

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        if abs_path is not None:
            default_path = os.path.join(abs_path, default_path)
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, len_dataset=None, config=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, len_dataset=len_dataset, config=config)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    skip_index = cfg['data']['skip_index']
    pose_name = cfg['data']['pose_name']
    extension = cfg['data']['extension']
    use_syncloss = cfg['training']['use_syncloss']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    
    # For nerf, we include three dataset types: 1. replica 2. llff 3. real scene
    # Now, the experiment is mainly on replica dataset.
    if dataset_type == "Replica":
        dataset = data.ReplicaDataset(dataset_folder, mode, cfg['data']['focal_length'])
    elif dataset_type == "ReplicaSemantic":
        semantic_label = cfg['data']['semantic_label']
        dataset = data.ReplicaSemanticDataset(dataset_folder, mode, cfg['data']['focal_length'], semantic_label=semantic_label)
    elif dataset_type == "SFM":
        dataset = data.SFMDataset(dataset_folder, mode, skip_index=skip_index, pose_name=pose_name)
    elif dataset_type == "lip_someone":
        dataset = data.SomeonesLipDataset(dataset_folder, mode, cfg=cfg, img_ext=extension, use_syncloss=use_syncloss)
    elif dataset_type == "lip_someone_mel":
        dataset = data.SomeonesLipMelDataset(dataset_folder, mode, cfg=cfg, img_ext=extension, use_syncloss=use_syncloss)        
    elif dataset_type == "lip_face_someone":
        face_dataset_folder = cfg['data']['face_dataset_folder']
        lip_dataset_folder = cfg['data']['lip_dataset_folder']
        use_syncloss = cfg['training']['use_syncloss']
        dataset = data.SomeonesLipFaceDataset(lip_dataset_folder=lip_dataset_folder, face_dataset_folder=face_dataset_folder, 
                                                mode=mode, img_ext=extension)
    elif dataset_type == "audio":
        dataset = data.LRS2Dataset(dataset_folder, mode, cfg=cfg)
 
    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'partial_pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        inputs_field = data.PartialPointCloudField(
            cfg['data']['pointcloud_file'], transform,
            multi_files= cfg['data']['multi_files']
        )
    elif input_type == 'pointcloud_crop':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
    
        inputs_field = data.PatchPointCloudField(
            cfg['data']['pointcloud_file'], 
            transform,
            multi_files= cfg['data']['multi_files'],
        )
    
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field

def set_logger(cfg):
    os.makedirs(cfg['training']['out_dir'], exist_ok=True)
    logfile = os.path.join(cfg['training']['out_dir'],
                           cfg['training']['logfile'])
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)
