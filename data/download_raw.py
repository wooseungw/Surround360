from datasets import load_dataset
from pathlib import Path

def download_raw_data():
    # Define the dataset name and the path to save the data
    dataset_name = "Silviase/QuIC-360"
    # Convert dataset_name to a suitable directory format by replacing slashes
    dataset_dir_name = dataset_name.replace("Silviase/", "").replace("-", "")
    data_dir = Path("data/raw") / dataset_dir_name
    # Create the directory if it doesn't exist
    if not data_dir.exists():
        # Create the directory structure
        print(f"Creating directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    ds = load_dataset(dataset_name)
    ds["train"].to_csv(data_dir/"train.csv", index=False)    
    ds["test"].to_csv(data_dir/"test.csv", index=False)
    ds["valid"].to_csv(data_dir/"valid.csv", index=False)
    ds["test_refer"].to_csv(data_dir/"test_refer.csv", index=False)

if __name__ == "__main__":

    download_raw_data()
    
