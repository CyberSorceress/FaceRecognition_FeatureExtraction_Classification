import kagglehub

def download_lfw_dataset():
    print("Downloading LFW dataset from KaggleHub...")
    path = kagglehub.dataset_download("atulanandjha/lfwpeople")
    print("Dataset downloaded to:", path)
    return path
