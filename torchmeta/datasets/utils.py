import os
import json
import requests
from tqdm import tqdm

# Make sure other utility functions are available
def get_asset(folder, filename):
    """
    Gets a file from the assets folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(current_dir), 'assets', folder)
    return os.path.join(assets_dir, filename)

def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

def download_file_from_google_drive(id, root, filename, md5=None):
    """Download a file from Google Drive.
    
    Args:
        id (str): Google Drive id for file
        root (str): Directory where the file will be saved
        filename (str): Name of the file to save
        md5 (str, optional): MD5 checksum for the file
    """
    # Make directory if it doesn't exist
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
        
    destination = os.path.join(root, filename)
    
    if os.path.exists(destination):
        return
    
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)
    
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        with tqdm(unit='B', unit_scale=True, desc=destination) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
