#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train \
		--train-src=./zh_en_data/train.zh \
		--train-tgt=./zh_en_data/train.en \
		--dev-src=./zh_en_data/dev.zh \
		--dev-tgt=./zh_en_data/dev.en \
		--vocab=vocab.json \
		--cuda \
		--amp \
		--pin-memory \
		--num-workers=2 \
		--lr=5e-4 \
		--patience=3 \
		--valid-niter=200 \
		--batch-size=32 \
		--embed-size=1024 \
		--hidden-size=768 \
		--dropout=.3 \
		--max-epoch=1
elif [ "$1" = "train_a100" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train \
		--train-src=./zh_en_data/train.zh \
		--train-tgt=./zh_en_data/train.en \
		--dev-src=./zh_en_data/dev.zh \
		--dev-tgt=./zh_en_data/dev.en \
		--vocab=vocab.json \
		--cuda \
		--amp \
		--pin-memory \
		--num-workers=4 \
		--lr=1e-3 \
		--patience=3 \
		--valid-niter=50 \
		--batch-size=128 \
		--embed-size=1024 \
		--hidden-size=768 \
		--dropout=.3 \
		--max-epoch=1
elif [ "$1" = "train_mps" ]; then
	python run.py train \
		--train-src=./zh_en_data/train.zh \
		--train-tgt=./zh_en_data/train.en \
		--dev-src=./zh_en_data/dev.zh \
		--dev-tgt=./zh_en_data/dev.en \
		--vocab=vocab.json \
		--mps \
		--num-workers=0 \
		--lr=5e-4 \
		--patience=3 \
		--valid-niter=200 \
		--batch-size=32 \
		--embed-size=1024 \
		--hidden-size=768 \
		--dropout=.3 \
		--max-epoch=1
elif [ "$1" = "train_toy_mps" ]; then
	python run.py train \
		--train-src=./zh_en_data/train.zh \
		--train-tgt=./zh_en_data/train.en \
		--dev-src=./zh_en_data/dev.zh \
		--dev-tgt=./zh_en_data/dev.en \
		--vocab=vocab.json \
		--mps \
		--num-workers=0 \
		--max-samples=1000 \
		--lr=5e-4 \
		--patience=3 \
		--valid-niter=50 \
		--batch-size=32 \
		--embed-size=256 \
		--hidden-size=256 \
		--dropout=.3 \
		--max-epoch=1
elif [ "$1" = "test" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./zh_en_data/test.zh ./zh_en_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "dev" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./zh_en_data/dev.zh ./zh_en_data/dev.en outputs/dev_outputs.txt --cuda
elif [ "$1" = "test_mps" ]; then
	python run.py decode model.bin ./zh_en_data/test.zh ./zh_en_data/test.en outputs/test_outputs.txt --mps
elif [ "$1" = "dev_mps" ]; then
	python run.py decode model.bin ./zh_en_data/dev.zh ./zh_en_data/dev.en outputs/dev_outputs.txt --mps
elif [ "$1" = "tensorboard" ]; then
	tensorboard --logdir runs --bind_all
else
	echo "Invalid Option Selected. Valid options: train, train_a100, train_mps, train_toy_mps, test, dev, test_mps, dev_mps, tensorboard"
fi
