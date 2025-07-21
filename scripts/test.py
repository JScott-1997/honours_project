import zipfile
import os

path_to_one_zip = "data/raw/300_P.zip"

with zipfile.ZipFile(path_to_one_zip, 'r') as archive:
    print("📦 Files in zip:")
    for f in archive.namelist():
        print(f)

    content = archive.read("300_TRANSCRIPT.csv").decode("latin-1")
    print("\n📝 First few lines of 300_TRANSCRIPT.csv:")
    print("\n".join(content.splitlines()[:5]))
