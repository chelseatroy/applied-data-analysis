"""
Download and unzip MovieLens 100K into the data/ directory.
Called automatically by compute scripts if data is not present.
"""
import urllib.request
import zipfile
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SENTINEL = os.path.join(DATA_DIR, "ml-100k", "u.data")


def download_if_needed():
    if os.path.exists(SENTINEL):
        return
    print("Downloading MovieLens 100K...")
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Done.")


if __name__ == "__main__":
    download_if_needed()
