import kagglehub

# Download latest version
path = kagglehub.dataset_download("nqa112/vietnamese-bike-and-motorbike")

print("Path to dataset files:", path)