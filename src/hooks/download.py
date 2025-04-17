import requests
import os
def Download_PDF(url, save_path="instance/temp.pdf", remove=False):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)

        if remove:
            # Ensure the file is removed after being returned
            try:
                os.remove(save_path)
            except OSError as e:
                print(f"Error deleting file: {e}")

        return save_path
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")