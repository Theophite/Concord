"""Quick sanity check: measure pretrained T5-small on CoLA val before
running the full transfer experiment.
"""
import sys
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset


def main():
    device = 'cuda'
    print("Loading T5-small + CoLA...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    ds = load_dataset('glue', 'cola')['validation']
    print(f"  CoLA val size: {len(ds)}")

    # T5's CoLA prefix: "cola sentence: <text>"
    # Targets: "acceptable" or "unacceptable"
    pos_token = 'acceptable'  # label = 1
    neg_token = 'unacceptable'  # label = 0
    pos_id = tokenizer(pos_token, return_tensors='pt')['input_ids'][0, 0].item()
    neg_id = tokenizer(neg_token, return_tensors='pt')['input_ids'][0, 0].item()
    print(f"  pos_id ({pos_token!r}) = {pos_id}  "
          f"neg_id ({neg_token!r}) = {neg_id}")

    inputs = [f"cola sentence: {x}" for x in ds['sentence']]
    labels = ds['label']
    enc = tokenizer(inputs, max_length=64, truncation=True,
                     padding='max_length', return_tensors='pt')

    model.eval()
    correct = total = 0
    bsz = 32
    n = enc['input_ids'].size(0)
    with torch.no_grad():
        for i in range(0, n, bsz):
            inp = enc['input_ids'][i:i+bsz].to(device)
            mask = enc['attention_mask'][i:i+bsz].to(device)
            lbl = torch.tensor(labels[i:i+bsz], device=device)
            decoder_inp = torch.full((inp.size(0), 1),
                                       model.config.decoder_start_token_id,
                                       device=device, dtype=torch.long)
            out = model(input_ids=inp, attention_mask=mask,
                         decoder_input_ids=decoder_inp)
            logits_first = out.logits[:, 0]
            pred_pos = logits_first[:, pos_id] > logits_first[:, neg_id]
            true_pos = (lbl == 1)
            correct += (pred_pos == true_pos).sum().item()
            total += inp.size(0)
    print(f"  CoLA val_acc (pretrained): {correct/total*100:.2f}%  "
          f"({correct}/{total})")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)
    main()
