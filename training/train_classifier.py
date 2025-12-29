import collections
import torch
from datasets import load_dataset #ignore 
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm


# ------------------ CONFIG ------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
OUTPUT_MODEL_PATH = "models/classifier/sentiment/best_model.pt"
TOKENIZER_DIR = "models/classifier/tokenizer"
# --------------------------------------------


def clean_text(sample):
    sample["text"] = sample["text"].strip()
    return sample


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )


def prepare_dataloaders(tokenizer):
    dataset = load_dataset("imdb")
    dataset = dataset.map(clean_text)
    dataset.pop("unsupervised")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_batch(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    device
):
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        model.eval()
        correct_preds = 0
        total_preds = 0
        total_eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"]
                )

                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)

                correct_preds += (predictions == batch["label"]).sum().item()
                total_preds += batch["label"].size(0)

        accuracy = correct_preds / total_preds
        avg_eval_loss = total_eval_loss / len(test_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            print("✅ Best model saved")

        print(f"Validation loss: {avg_eval_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_DIR)

    train_loader, test_loader = prepare_dataloaders(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )


if __name__ == "__main__":
    main()
