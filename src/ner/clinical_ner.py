"""
Modelo de NER Clinico baseado em Transformer (BERTimbau / BioBERTpt).

Implementa fine-tuning de modelos pre-treinados em portugues para
reconhecimento de entidades clinicas em textos medicos brasileiros.
Suporta treinamento, avaliacao e inferencia.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset

from src.ner.entity_types import BIO_LABELS, LABEL2ID, ID2LABEL


class ClinicalNERModel:
    """
    Modelo de Named Entity Recognition para textos clinicos em PT-BR.

    Encapsula o ciclo completo de NER clinico:
    - Carregamento de modelos pre-treinados (BERTimbau, BioBERTpt)
    - Fine-tuning com dados anotados no formato BIO
    - Inferencia com pos-processamento de entidades
    - Serializacao e carregamento de modelos treinados

    Attributes:
        model_name: Nome ou path do modelo Hugging Face
        device: Dispositivo de computacao (cpu/cuda/mps)
        max_length: Comprimento maximo de sequencia em tokens
        tokenizer: Tokenizer do modelo
        model: Modelo de classificacao de tokens

    Example:
        >>> ner = ClinicalNERModel("neuralmind/bert-base-portuguese-cased")
        >>> ner.load_model()
        >>> entities = ner.predict("Paciente com HAS em uso de Losartana 50mg")
        >>> print(entities)
        [
            {"text": "HAS", "label": "CONDICAO", "start": 14, "end": 17, "score": 0.97},
            {"text": "Losartana", "label": "MEDICAMENTO", "start": 28, "end": 37, "score": 0.99},
            {"text": "50mg", "label": "DOSAGEM", "start": 38, "end": 42, "score": 0.95}
        ]
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", "./data/models")

        # Detectar dispositivo automaticamente
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        logger.info(
            f"ClinicalNERModel inicializado | modelo={model_name} | "
            f"device={self.device} | max_length={max_length}"
        )

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Carrega modelo e tokenizer.

        Args:
            checkpoint_path: Path para checkpoint local (fine-tunado).
                Se None, carrega o modelo pre-treinado base.
        """
        load_from = checkpoint_path or self.model_name

        logger.info(f"Carregando modelo de {load_from}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            cache_dir=self.cache_dir,
            use_fast=True,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            load_from,
            num_labels=len(BIO_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            cache_dir=self.cache_dir,
        )

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Modelo carregado | parametros={param_count:,} | "
            f"labels={len(BIO_LABELS)} | device={self.device}"
        )

    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        aggregate: bool = True,
    ) -> List[Dict]:
        """
        Extrai entidades clinicas de um texto.

        Args:
            text: Texto clinico em portugues para analise
            threshold: Score minimo para considerar uma entidade
            aggregate: Se True, agrupa tokens B-/I- em entidades completas

        Returns:
            Lista de dicionarios com entidades encontradas:
            [{"text": str, "label": str, "start": int, "end": int, "score": float}]
        """
        if not self._is_loaded:
            raise RuntimeError("Modelo nao carregado. Execute load_model() primeiro.")

        # Tokenizar
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        # Mover para device
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Processar logits
        predictions = torch.softmax(outputs.logits, dim=-1)
        pred_labels = torch.argmax(predictions, dim=-1)[0].cpu().numpy()
        pred_scores = predictions[0].cpu().numpy()

        # Converter para entidades
        raw_entities = []
        for idx, (label_id, offsets) in enumerate(zip(pred_labels, offset_mapping)):
            if offsets == (0, 0):  # tokens especiais [CLS], [SEP], padding
                continue

            label = ID2LABEL[label_id]
            score = float(pred_scores[idx][label_id])

            if label != "O" and score >= threshold:
                start, end = offsets
                raw_entities.append({
                    "text": text[start:end],
                    "label": label,
                    "start": start,
                    "end": end,
                    "score": score,
                    "bio_tag": label[:2],  # B- ou I-
                    "entity_type": label[2:] if len(label) > 2 else label,
                })

        if aggregate:
            return self._aggregate_entities(raw_entities, text)

        return raw_entities

    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        batch_size: int = 16,
    ) -> List[List[Dict]]:
        """
        Extrai entidades de multiplos textos em lote.

        Args:
            texts: Lista de textos clinicos
            threshold: Score minimo
            batch_size: Tamanho do lote para inferencia

        Returns:
            Lista de listas de entidades, uma por texto
        """
        if not self._is_loaded:
            raise RuntimeError("Modelo nao carregado. Execute load_model() primeiro.")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = [self.predict(text, threshold) for text in batch]
            results.extend(batch_results)

        return results

    def _aggregate_entities(
        self, raw_entities: List[Dict], original_text: str
    ) -> List[Dict]:
        """
        Agrupa tokens B-/I- consecutivos do mesmo tipo em entidades completas.

        Exemplo:
            B-MEDICAMENTO("Dipirona") + I-MEDICAMENTO("sodica") ->
            MEDICAMENTO("Dipirona sodica")
        """
        if not raw_entities:
            return []

        aggregated = []
        current = None

        for entity in raw_entities:
            bio_tag = entity["bio_tag"]
            entity_type = entity["entity_type"]

            if bio_tag == "B-":
                # Salvar entidade anterior se existir
                if current:
                    current["text"] = original_text[current["start"]:current["end"]]
                    aggregated.append(current)

                # Iniciar nova entidade
                current = {
                    "text": entity["text"],
                    "label": entity_type,
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity["score"],
                }

            elif bio_tag == "I-" and current and current["label"] == entity_type:
                # Continuar entidade existente
                current["end"] = entity["end"]
                current["score"] = min(current["score"], entity["score"])

            else:
                # Tag I- sem B- correspondente — tratar como B-
                if current:
                    current["text"] = original_text[current["start"]:current["end"]]
                    aggregated.append(current)

                current = {
                    "text": entity["text"],
                    "label": entity_type,
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity["score"],
                }

        # Ultima entidade
        if current:
            current["text"] = original_text[current["start"]:current["end"]]
            aggregated.append(current)

        return aggregated

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./data/models/clinical-ner-ptbr",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3,
    ) -> Dict:
        """
        Fine-tuna o modelo com dados clinicos anotados.

        Args:
            train_dataset: Dataset de treino no formato HF
            eval_dataset: Dataset de validacao (opcional)
            output_dir: Diretorio para salvar checkpoints
            epochs: Numero de epocas de treinamento
            batch_size: Tamanho do batch
            learning_rate: Taxa de aprendizado
            warmup_ratio: Proporcao de warmup
            weight_decay: Peso de regularizacao L2
            early_stopping_patience: Paciencia para early stopping

        Returns:
            Dicionario com metricas de treinamento
        """
        if not self._is_loaded:
            self.load_model()

        self.model.train()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
        )

        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics if eval_dataset else None,
            callbacks=callbacks,
        )

        logger.info(
            f"Iniciando fine-tuning | epochs={epochs} | batch={batch_size} | "
            f"lr={learning_rate} | dataset_size={len(train_dataset)}"
        )

        train_result = trainer.train()

        # Salvar modelo final
        trainer.save_model(f"{output_dir}/final")
        self.tokenizer.save_pretrained(f"{output_dir}/final")

        metrics = train_result.metrics
        logger.info(f"Treinamento concluido | loss={metrics.get('train_loss', 0):.4f}")

        return metrics

    def _compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Calcula metricas de NER (precision, recall, F1) usando seqeval."""
        from seqeval.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
        )

        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)

        # Converter IDs para labels, ignorando -100 (padding)
        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != -100:
                    true_seq.append(ID2LABEL[label_id])
                    pred_seq_labels.append(ID2LABEL[pred_id])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_labels)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

    def save(self, path: str) -> None:
        """Salva modelo e tokenizer em disco."""
        if not self._is_loaded:
            raise RuntimeError("Modelo nao carregado.")

        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Modelo salvo em {path}")

    def load(self, path: str) -> None:
        """Carrega modelo previamente salvo."""
        self.load_model(checkpoint_path=path)
