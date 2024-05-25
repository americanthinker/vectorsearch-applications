import sys

sys.path.append("../")
try:
    from src.preprocessor.preprocessing import FileIO
except ModuleNotFoundError:
    from preprocessing import FileIO

from torch import cuda
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, models


class FineTuner:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.model_name = model_name

    def finetune(
        data_path: str = "../data/qa_training_triplets.json",
        num_epochs: int = 3,
        chunk_size: int = 16,
    ):
        data = FileIO.load_json(data_path)

        train_examples = [
            InputExample(
                texts=[sample["anchor"], sample["positive"], sample["hard_negative"]]
            )
            for sample in data
        ]
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=32,
        )

        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        warmup_steps = int(
            len(train_dataloader) * num_epochs * 0.1
        )  # 10% of train data

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
        )

        self.model.save(
            path=f"../models/{self.model_name}-finetuned-500",
            model_name=f"{self.model_name}-finetuned-500",
        )
