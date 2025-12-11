import os
import csv
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

LABEL2ID = {"ENTAILMENT": 0, "NONE": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item


def prepare_dataset(tokenizer, split, max_length):
    data = list(split)

    premise = [x['premise'] for x in data]
    hypothesis = [x['hypothesis'] for x in data]

    enc = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

    # Mapear 0 -> 'ENTAILMENT', 1 -> 'NONE'
    int2label = {0: "ENTAILMENT", 1: "NONE"}
    labels = [LABEL2ID[int2label[x['entailment_judgment']]] for x in data]
    return enc, labels

def make_loader(tokenizer, split, max_length, batch_size, shuffle=False):
    enc, labels = prepare_dataset(tokenizer, split, max_length)
    ds = NLIDataset(enc, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def eval_loop(model, dataloader, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro")
    loss = total_loss / len(dataloader)
    return loss, acc, macro_f1

def run_training(cfg, run_id):
    seed = cfg['seed'] + run_id
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTreino {run_id+1}/{cfg['runs']} usando {device}")

    ds = load_dataset("nilc-nlp/assin2")
    train = ds["train"].shuffle(seed=seed)
    val = ds["validation"].shuffle(seed=seed)
    test = ds["test"].shuffle(seed=seed)

    counts = Counter(str(ex['entailment_judgment']).upper() for ex in train)
    print("Labels disponíveis no train:", counts)

    Dl = train
    print(f"Dataset inicial criado com {len(Dl)} amostras")

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    dl_train = make_loader(tokenizer, Dl, cfg['max_length'], cfg['batch_size'], shuffle=True)
    dl_val = make_loader(tokenizer, val, cfg['max_length'], cfg['batch_size'])
    dl_test = make_loader(tokenizer, test, cfg['max_length'], cfg['batch_size'])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'],
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg['lr']))

    for epoch in range(cfg['epochs_init']):
        model.train()
        total_loss = 0
        for batch in tqdm(dl_train, desc=f"Run {run_id+1} - Epoch {epoch+1}/{cfg['epochs_init']}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
        print(f"Loss epoch {epoch+1}: {total_loss/len(dl_train):.4f}")

    val_loss, val_acc, val_f1 = eval_loop(model, dl_val, device)
    test_loss, test_acc, test_f1 = eval_loop(model, dl_test, device)

    print(f"Run {run_id+1} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    return {"run": run_id+1, "val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1}

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    all_results = []

    os.makedirs("outputs", exist_ok=True)
    csv_path = "outputs/resultados_assin2.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Val_Accuracy", "Val_F1", "Test_Accuracy", "Test_F1"])

    for run_id in range(cfg['runs']):
        result = run_training(cfg, run_id)
        all_results.append(result)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([result["run"], result["val_acc"], result["val_f1"], result["test_acc"], result["test_f1"]])

    print("\n📊 RESULTADOS FINAIS:")
    for r in all_results:
        print(f"Run {r['run']}: Val Acc={r['val_acc']:.4f}, Val F1={r['val_f1']:.4f}")

    val_mean = np.mean([r["val_acc"] for r in all_results])
    f1_mean = np.mean([r["val_f1"] for r in all_results])
    print(f"\n📈 Médias finais — Val Acc: {val_mean:.4f}, Val F1: {f1_mean:.4f}")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Médias", val_mean, f1_mean, "-", "-"])

    print(f"\n CSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
