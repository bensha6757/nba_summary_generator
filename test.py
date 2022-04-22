import torch
import transformers
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

import slurm
import util
from options import Options
import data
import model
from datasets import load_metric


def evaluate(model, dataset, dataloader, tokenizer, opt):
    predictions = []
    references = []
    model = model.module if hasattr(model, 'module') else model
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt' % opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, description_ids, description_mask) = batch

            outputs = model.generate(
                input_ids=description_ids.cuda(),
                attention_mask=description_mask.cuda(),
                max_length=700,
                # num_beams=5
            )

            for k, output in enumerate(outputs):
                ans = tokenizer.decode(output, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                predictions.append(ans)
                references.append(example['summary'])

                if opt.write_results:
                    fw.write(str(example['id']) + "\tprediction:\n" + ans + '\n')
                    fw.write(str(example['id']) + "\treference:\n" + example['summary'] + '\n')
                    fw.write('************************************************\n\n\n')

                if (i + 1) % opt.eval_print_freq == 0:
                    print(str(example['id']) + "\tprediction:\n" + ans + '\n')
                    print(str(example['id']) + "\treference:\n" + example['summary'] + '\n')
                    print('************************************************\n\n\n')

        bertscore_metric = load_metric('bertscore')
        bert_scores = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            lang="en")
        print('BERT scores: ' + bert_scores)
        print('F1 BERT scores: ' + bert_scores['f1'])


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    slurm.init_distributed_mode(opt)
    opt.train_batch_size = opt.batch_size_per_gpu * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small', return_dict=False)

    collator_function = data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        # use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = data.Dataset(
        eval_examples,
        opt.n_descriptions,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.batch_size_per_gpu,
        num_workers=20,
        collate_fn=collator_function
    )

    model_class = model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        util.save_distributed_dataset(eval_dataset.data, opt)
