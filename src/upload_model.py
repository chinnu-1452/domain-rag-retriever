from huggingface_hub import HfApi, login

login()
api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/hr-bge-small-ft"
api.upload_folder(folder_path="hr-bge-small-ft", repo_id=repo)
print("Uploaded:", repo)