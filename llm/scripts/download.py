from typing import Optional


def download_from_hub(repo_id: Optional[str] = None, 
                      local_dir: str = None, 
                      token: str = None) -> None:

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id,
                      local_dir=local_dir,
                      token=token,
                      local_dir_use_symlinks=False)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
