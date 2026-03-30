import requests
from tqdm import tqdm
from certgnn.utils import load_config, get_project_root


def download_dataset():
    print("Fetching file list from API...")

    cfg = load_config()
    root = get_project_root()

    data_dir = root / cfg['paths']['raw_dir']
    article_id = cfg['download']['article_id']
    target_files = cfg['download']['target_files']

    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"

    print("Fetching file list from API...")
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    data_dir.mkdir(parents=True, exist_ok=True)

    for file_info in files:
        filename = file_info['name']

        if filename not in target_files:
            continue

        download_url = file_info['download_url']
        file_path = data_dir / filename
        file_size = file_info['size']
        size_gb = file_size / (1024**3)

        # Check if the file already exists and is fully downloaded
        if file_path.exists():
            existing_size = file_path.stat().st_size
            if existing_size == file_size:
                print(f"File {filename} already exists and is fully downloaded. Skipping.")
                continue
            else:
                print(f"File {filename} is incomplete. Resuming/Restarting download...")

        print(f"Downloading {filename} ({size_gb:.2f} GB)...")

        # Stream the download
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=file_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    print("Download complete!")


if __name__ == "__main__":
    download_dataset()
