from .business import run as run_business
from .data_understanding import run as run_data_understanding
from .data_preparation import run as run_data_preparation
from .modeling import run as run_modeling
from .evaluation import run as run_evaluation
from .deployment import run as run_deployment

def main():
    print("Pipeline CRISP-DM:\n")
    run_business()
    run_data_understanding()
    run_data_preparation()
    run_modeling()
    run_evaluation()
    run_deployment()

if __name__ == "__main__":
    main()
