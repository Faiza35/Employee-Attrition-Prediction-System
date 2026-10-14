"""Run the whole pipeline in order: data -> EDA -> training."""
from generate_data import generate_dataset
from eda import run_eda
from preprocessing import clean_data, load_data
from train_model import train_and_compare
from utils import DATA_PATH

if __name__ == "__main__":
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_PATH, index=False)
        print(f"Generated dataset at {DATA_PATH}")
    run_eda(clean_data(load_data()))
    train_and_compare()
