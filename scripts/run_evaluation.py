import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR, ENV_VARIABLES


from agents.evaluation import DataLoader, Evaluator
from agents.rag.base import RAGAgent

async def main():
    # Load the data
    # test_data = DATA_DIR / "data_to_process" / "test.jsonl"
    # data_loader = DataLoader(file_dir=test_data)

    validation_data = DATA_DIR / "golden_dataset.json"
    data_loader = DataLoader(file_dir=validation_data)

    # Initialize the assistant
    assistant = RAGAgent(llm_provider="azure", retrieval_args={})

    # Initialize the evaluator
    evaluator = Evaluator(dataloader=data_loader, model_pipeline=assistant)

    # Run the evaluation
    sampled_data = await evaluator.run_prediction(size_sample=300)

    evaluations_run_dir = DATA_DIR / "evaluations"
    if not evaluations_run_dir.exists():
        evaluations_run_dir.mkdir(parents=True, exist_ok=True)
    # list runs folders
    runs_folders = evaluations_run_dir.glob("run_*")
    run_id = len(list(runs_folders)) + 1
    run_dir = evaluations_run_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save the results
    with open(run_dir / "predictions_results.json", "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=4)

    # Run the evaluation
    #sampled_data = await evaluator.evaluate_prediction(sampled_data)

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())