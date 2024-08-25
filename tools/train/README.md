# Training
If you need to test that training is working you can perform the following

- mock_pretrain
_ mock_finetune

## mock pretrain
this will perform a pretrain against some random generated data.
this will not do any useful training but show that the training code works.

```bash
python tools/train/mock_pretrain_train.py
```

## mock finetune
this will perform a pretrain against some random generated data.
this will not do any useful training but show that the training code works.

```bash
python tools/train/mock_finetune_train.py
```

## visalize schedulers
this will produce a pretty graph allowing you to visualize training schedulers

```bash
python tools/train/visualize_schedulers.py --scheduler warmup --total_steps 1000 --initial_lr 0.00002 --warmup_steps 100
```

or

```bash
python tools/train/visualize_schedulers.py --scheduler cosine_annealing --total_steps 1000 --initial_lr 0.1 --min_lr 0.01
```
