# token sequence가 주어졌을 때, 바로 다음에 나올 token을 예측하는 모델
import math

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import build_vocab_from_iterator

from module import TransformerModel


def data_process(tokenizer, vocab, raw_text_iter):
    data = [
        torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, batch_size):

    assert data.ndim == 1, data.shape

    # Divide the dataset into bsz parts.
    num_batch = data.numel() // batch_size

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[: num_batch * batch_size]

    # Evenly divide the data across the bsz batches.
    data = data.reshape(batch_size, -1).t().contiguous()

    return data


def get_batch(bptt, source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target


class Agent:
    def __init__(self):
        # device configuration
        self.device = torch.device("cuda")

        # Download wikitext-2
        print("\n----- Download wikitext-2 -----")
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
        test_filepath, val_filepath, train_filepath = extract_archive(download_from_url(url))
        print("train_filepath: ", train_filepath)
        print("val_filepath: ", val_filepath)
        print("test_filepath: ", test_filepath)

        # Build vocab using train dataset
        print("\n----- Build vocab using train dataset -----")
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            map(tokenizer, iter(open(train_filepath, encoding="utf8")))
        )

        # 각 데이터셋마다 모든 문장들을 "토큰화 -> 정수 인코딩" 후 일렬로 나열.
        print("\n----- 각 데이터셋마다 문장들에 대해 (토큰화 -> 정수 인코딩)을 거치고, concatenation -----")
        train_data = data_process(tokenizer, vocab, iter(open(train_filepath, encoding="utf8")))
        val_data = data_process(tokenizer, vocab, iter(open(val_filepath, encoding="utf8")))
        test_data = data_process(tokenizer, vocab, iter(open(test_filepath, encoding="utf8")))
        print("train_data_shape: ", train_data.shape, ",", train_data.dtype)
        print("val_data_shape: ", val_data.shape, ",", val_data.dtype)
        print("test_data_shape: ", test_data.shape, ",", test_data.dtype)

        # Batchfiy
        print("\n----- Batchfiy (나누어 떨어지지 않으면 버림) ------")
        train_batch_size = 20
        eval_batch_size = 10
        train_data = batchify(train_data, train_batch_size).to(self.device)
        val_data = batchify(val_data, eval_batch_size).to(self.device)
        test_data = batchify(test_data, eval_batch_size).to(self.device)
        print("train_data_shape: ", train_data.shape, ",", train_data.dtype)
        print("val_data_shape: ", val_data.shape, ",", val_data.dtype)
        print("test_data_shape: ", test_data.shape, ",", test_data.dtype)

        self.train_data = train_data  # (?, B)
        self.val_data = val_data  # (?, B)
        self.test_data = test_data  # (?, B)

        # Network
        self.net = TransformerModel(
            vocab_size=len(vocab),
            d_model=200,
            n_heads=2,
            dim_feedforward=200,
            n_transformer_layers=2,
            dropout=0.2,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=5.0)

        # ETC
        self.num_epochs = 3  # The number of epochs
        self.epoch = 0

        self.bptt = 35
        self.vocab_size = len(vocab)

    def train(self):
        start_epoch = self.epoch + 1
        for self.epoch in range(start_epoch, self.num_epochs + 1):
            # train step
            print("\n-------------------------------------------------------------------------")
            print(f"\nEpoch {self.epoch} - LR {self.optimizer.param_groups[0]['lr']}")
            self._train_epoch()
            val_loss = self._validate_epoch(self.val_data)

            print("-" * 89)
            print(
                f"| end of epoch {self.epoch:3d} "
                f"| valid loss {val_loss:5.2f} "
                f"| valid ppl {math.exp(val_loss):8.2f}"
            )
            print("-" * 89)

    def _train_epoch(self):
        self.net.train()
        total_loss = 0.0
        log_interval = 200
        src_mask = self.net.generate_square_subsequent_mask(self.bptt).to(self.device)

        for step, i in enumerate(range(0, self.train_data.shape[0] - 1, self.bptt)):
            data, targets = get_batch(
                self.bptt, self.train_data, i
            )  # data: (bptt, B), targets: (bptt, B)

            if data.shape[0] != self.bptt:
                src_mask = self.net.generate_square_subsequent_mask(data.shape[0]).to(self.device)
            output = self.net(data, src_mask)  # (bptt, B, vocab_size)
            loss = self.criterion(output.view(-1, self.vocab_size), targets.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            if step % log_interval == 0 and step > 0:
                cur_loss = total_loss / log_interval
                print(
                    f"| epoch {self.epoch:3d} "
                    f"| {step:5d}/{len(self.train_data) // self.bptt:5d} batches "
                    f"| loss {cur_loss:5.2f} "
                    f"| ppl {math.exp(cur_loss):8.2f}"
                )
                total_loss = 0

    @torch.no_grad()
    def _validate_epoch(self, data_source):
        self.net.eval()
        total_loss = 0.0
        src_mask = self.net.generate_square_subsequent_mask(self.bptt).to(self.device)

        for i in range(0, data_source.size(0) - 1, self.bptt):
            data, targets = get_batch(self.bptt, data_source, i)
            if data.shape[0] != self.bptt:
                src_mask = self.net.generate_square_subsequent_mask(data.shape[0]).to(self.device)
            output = self.net(data, src_mask)
            output_flat = output.view(-1, self.vocab_size)
            total_loss += len(data) * self.criterion(output_flat, targets.view(-1)).item()

        return total_loss / (len(data_source) - 1)


def main():
    agent = Agent()
    agent.train()


if __name__ == "__main__":
    main()
