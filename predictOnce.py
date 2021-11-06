import os
import numpy as np
import torch
from transformers import BertTokenizer
from model import JointBERT


class Estimator:
    def __init__(self, args):
        self.intent_label_lst = [label.strip() for label in open(args.intent_label_file, 'r', encoding='utf-8')]
        self.slot_label_lst = [label.strip() for label in open(args.slot_label_file, 'r', encoding='utf-8')]

        # Check whether model exists
        if not os.path.exists(args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.model = JointBERT.from_pretrained(args.model_dir,
                                               args=args,
                                               intent_label_lst=self.intent_label_lst,
                                               slot_label_lst=self.slot_label_lst)
        self.model.to(args.device)
        self.model.eval()
        self.args = args

    def convert_input_to_tensor_data(self, input, tokenizer, pad_token_label_id,
                                     cls_token_segment_id=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        slot_label_mask = []

        words = list(input)
        tokens = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.args.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(self.args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        # Change to Tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        slot_label_mask = torch.tensor([slot_label_mask], dtype=torch.long)

        data = [input_ids, attention_mask, token_type_ids, slot_label_mask]

        return data

    def predict(self, input):
        # Convert input file to TensorDataset
        pad_token_label_id = self.args.ignore_index
        tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)
        batch = self.convert_input_to_tensor_data(input, tokenizer, pad_token_label_id)

        # Predict
        batch = tuple(t.to(self.args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            intent_pred = intent_logits.detach().cpu().numpy()

            # Slot prediction
            if self.args.use_crf:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = np.array(self.model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu().numpy()
            all_slot_label_mask = batch[3].detach().cpu().numpy()

        intent_pred = np.argmax(intent_pred, axis=1)[0]

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = []

        for i in range(slot_preds.shape[1]):
            if all_slot_label_mask[0, i] != pad_token_label_id:
                slot_preds_list.append(slot_label_map[slot_preds[0][i]])

        words = list(input)
        slots = list()
        slot = str()
        for i in range(len(words)):
            if slot_preds_list[i] == 'O':
                if slot == '':
                    continue
                slots.append({slot_preds_list[i - 1].split('-')[1]: slot})
                slot = str()
            else:
                slot += words[i]
        if slot != '':
            slots.append({slot_preds_list[len(words) - 1].split('-')[1]: slot})
        return self.intent_label_lst[intent_pred], slots


class Args:
    adam_epsilon = 1e-08
    batch_size = 16
    data_dir = 'data'
    device = 'cpu'
    do_eval = True
    do_train = False
    dropout_rate = 0.1
    eval_batch_size = 64
    gradient_accumulation_steps = 1
    ignore_index = 0
    intent_label_file = 'data/book/intent_label.txt'
    learning_rate = 5e-05
    logging_steps = 50
    max_grad_norm = 1.0
    max_seq_len = 50
    max_steps = -1
    model_dir = 'book_model'
    model_name_or_path = 'bert-base-chinese'
    model_type = 'bert-chinese'
    no_cuda = False
    num_train_epochs = 5.0
    save_steps = 200
    seed = 1234
    slot_label_file = 'data/book/slot_label.txt'
    slot_loss_coef = 1.0
    slot_pad_label = 'PAD'
    task = 'book'
    train_batch_size = 32
    use_crf = False
    warmup_steps = 0
    weight_decay = 0.0


if __name__ == "__main__":
    e = Estimator(Args)
    print(e.predict("哎我就是胡说八道就是玩"))
    print(e.predict("帮我查一下有没有三国演义吧"))
