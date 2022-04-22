import torch
import transformers
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from options import Options

import slurm
import util
import data
import model
from datasets import load_metric
from rouge import Rouge


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed)  # different seed for different sampling depending on global_rank
    train_sampler = DistributedSampler(train_dataset) if opt.is_distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.batch_size_per_gpu,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    model.train()
    while step < opt.total_steps:
        epoch += 1
        print('epoch #', epoch)
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, description_ids, description_mask) = batch

            train_loss = model(
                input_ids=description_ids.cuda(),
                attention_mask=description_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                    util.save(model, optimizer, scheduler, step, best_dev_em,
                              opt, checkpoint_path, 'best_dev')
                log = f"{step} / {opt.total_steps} |"
                log += f"train: {curr_loss / opt.eval_freq:.3f} |"
                log += f"evaluation: {100 * dev_em:.2f}EM |"
                log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                logger.info(log)
                if tb_logger is not None:
                    tb_logger.add_scalar("Evaluation", dev_em, step)
                    tb_logger.add_scalar("Training", curr_loss / opt.eval_freq, step)
                curr_loss = 0.

            if step % opt.save_freq == 0:
                util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break


def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.batch_size_per_gpu,
                            drop_last=False,
                            num_workers=10,
                            collate_fn=collator)
    model.eval()
    predictions = []
    references = []
    model = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, description_ids, description_mask) = batch

            outputs = model.generate(
                input_ids=description_ids.cuda(),
                attention_mask=description_mask.cuda(),
                max_length=700)

            for k, output in enumerate(outputs):
                ans = tokenizer.decode(output, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['summary']

                predictions.append(ans)
                references.append(gold)

                if (i + 1) % opt.eval_print_freq == 0:
                    print("\tprediction:\n" + ans + '\n')
                    print("\treference:\n" + gold + '\n')
                    print('************************************************\n\n\n')
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references)
        scores = [score['rouge-1']['f'] for score in scores]
        f1_score = sum(scores) / len(scores)
        print('F1 rouge score: ' + str(f1_score))
        # bertscore_metric = load_metric('bertscore')
        # bert_scores = bertscore_metric.compute(
        #     predictions=predictions,
        #     references=references,
        #     lang="en")
        # print('BERT scores: ' + bert_scores)
        # print('F1 BERT scores: ' + bert_scores['f1'])
    return f1_score
    # return bert_scores['f1']


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    # opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    opt.is_main = True
    # if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    # checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = data.Dataset(train_examples, opt.n_descriptions)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = data.Dataset(eval_examples, opt.n_descriptions)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
