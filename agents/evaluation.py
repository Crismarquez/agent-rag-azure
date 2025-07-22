from typing import Dict, Tuple
from pathlib import Path
import json
from datetime import datetime
import random
import asyncio
import time

from agents.rag.base import BaseAgent
from schemas.conversation import MessageItem

METADATA = {
    "user_id": "1",
    "conversation_id": "1"
}

class DataLoader:
    def __init__(self, file_dir: Path) -> None:
        self.file_dir = file_dir

    def load_data_jsonl(self) -> list:
        with open(self.file_dir, 'r') as archivo:
            dataset = [json.loads(line) for line in archivo]

        return dataset

    def load_data(self) -> list:
        with open(self.file_dir, 'r', encoding="utf-8") as archivo:
            dataset = json.load(archivo)

        for session in dataset:
            session["messages"] = self._conversation_format(session)

        return dataset

    def _conversation_format(self, session: dict) -> list:
        return [MessageItem(role="user", content=session["question"])]


class Evaluator:
    def __init__(self, dataloader, model_pipeline: BaseAgent):

        self.dataloader = dataloader
        self.assistant = model_pipeline

    async def run_prediction(self, size_sample: int = 100) -> None:
        dataset = self.dataloader.load_data()
        if len(dataset) >= size_sample:
            sample_size = size_sample
        else:
            sample_size = len(dataset)
            print(f"La lista tiene menos de {size_sample} elementos. Seleccionando todos los {sample_size} elementos.")

        sampled_data = random.sample(dataset, sample_size)
        # sampled_data = dataset[:10]

        async def evaluate_batch(batch):
            tasks = [self.assistant.run(session["messages"], METADATA) for session in batch]
            results = await asyncio.gather(*tasks)
            for session, result in zip(batch, results):
                session["result"] = {
                    "answer": result["messages"][-1].content,
                    "ids_content": result["ids_content"]
                }
                session.pop("messages")


        # # Dividir sampled_data en lotes de 4
        batches = [sampled_data[i:i + 4] for i in range(0, len(sampled_data), 4)]
        
        # Evaluar cada lote de forma secuencial (puedes también hacer esto concurrente si es necesario)
        for batch in batches:
            #time.sleep(1)  # for limit quota
            try:
                await evaluate_batch(batch)
            except Exception as e:
                print(f"Error evaluating batch: {e}")
                print("Assuming quota limit reached. Sleeping for 1 minute.")
                time.sleep(60)
                await evaluate_batch(batch)

        # for session in sampled_data:
        #     try:
        #         result = await self.assistant.run(session["messages"], METADATA)
        #        session["result"] = {
        #            "answer": result["messages"][-1].content,
        #            "ids_content": result["ids_content"]}
        #        session.pop("messages")
        #    except Exception as e:
        #        print(f"Error evaluating session: {e}")
        #        print("parsing error")
        #        continue

        return sampled_data
    
    async def evaluate_prediction(self, sampled_data: list) -> None:
        pass

