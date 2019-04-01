#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import math
import random
import time

import torch

from copy import deepcopy
from functools import reduce

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, bleu, tokenizer
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary

from mlperf_compliance import mlperf_log

def main(args, init_distributed=False):
    import_user_module(args)

    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)

    from mlperf_compliance.mlperf_log import transformer_print
    transformer_print(key=mlperf_log.RUN_CLEAR_CACHES) #before this tag we should run clearing caches on the host

    transformer_print(key=mlperf_log.OPT_NAME, value=args.optimizer)
    transformer_print(key=mlperf_log.OPT_LR, value=args.lr)
    transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=eval(args.adam_betas)[0])
    transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=eval(args.adam_betas)[1])
    transformer_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.adam_eps)

    torch.manual_seed(args.seed)
    transformer_print(key=mlperf_log.RUN_SET_RANDOM_SEED, value=args.seed)


    transformer_print(key=mlperf_log.RUN_START)
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    transformer_print(key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
            value={'alpha':args.lenpen,
                'beam_size':args.beam,
                'extra_decode_length':args.max_len_b,
                'vocab_size':task.target_dictionary.__len__()
                }
            )

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Initialize distributed training (after data loading)
    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('train').get_dummy_batch(1, max_positions)

    # Build trainer
    if (args.online_eval or args.target_bleu) and not args.remove_bpe:
        args.remove_bpe='@@ '

    device_name = 'CPU' if args.cpu else 'GPU'
    print('| training on {} {}s'.format(args.distributed_world_size, device_name))
    print('| max tokens per {} = {} and max sentences per {} = {}'.format(
        device_name,
        args.max_tokens,
        device_name,
        args.max_sentences,
    ))

    transformer_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.max_tokens)
    transformer_print(key=mlperf_log.INPUT_ORDER)

    trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    tgt_bleu = args.target_bleu or math.inf
    current_bleu = 0.0
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    transformer_print(key=mlperf_log.TRAIN_LOOP)

    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update and current_bleu < tgt_bleu:
        # train for one epoch
        transformer_print(key=mlperf_log.TRAIN_EPOCH, value=epoch_itr.epoch)

        start = time.time()
        train(args, trainer, task, epoch_itr)
        print("epoch time ", time.time() - start)

        start = time.time()
        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

         # Eval BLEU score
        transformer_print(key=mlperf_log.EVAL_START, value=epoch_itr.epoch)
        if args.online_eval or (not tgt_bleu is math.inf):
            print("begin eval")
            current_bleu = score(args, trainer, task, epoch_itr, args.gen_subset)
            transformer_print(key=mlperf_log.EVAL_ACCURACY, value={'epoch':epoch_itr.epoch, 'value':current_bleu})
            transformer_print(key=mlperf_log.EVAL_TARGET, value=tgt_bleu)
        transformer_print(key=mlperf_log.EVAL_STOP, value=epoch_itr.epoch)
        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    transformer_print(key=mlperf_log.RUN_STOP)
    transformer_print(key=mlperf_log.RUN_FINAL)
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
            if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break
        '''
        if i==1:
            break
        '''
    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats['loss'].avg)
    return valid_losses

def score(args, trainer, task, epoch_itr, subset):

    begin = time.time()

    if not subset in task.datasets.keys():
        task.load_dataset(subset)

    src_dict = deepcopy(task.source_dictionary) # This is necessary, generation of translations
    tgt_dict = deepcopy(task.target_dictionary) # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    align_dict = utils.load_align_dict(args.replace_unk)
    #mlperf_log.transformer_print(key=mlperf_log.EVAL_SIZE, value=task.dataset(subset).__len__())
    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=8,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    print(args.beam)
    generator = SequenceGenerator(tgt_dict, beam_size=args.beam,
        max_len_a=args.max_len_a, max_len_b=args.max_len_b, min_len=args.min_len,
        stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk,
        )
    # Generate and compute BLEU
    dict = dictionary.Dictionary()
    scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
    num_sentences = 0
    has_target = True
    if args.log_translations:
        log = open(os.path.join(args.save_dir, 'translations_epoch{}_{}'.format(epoch_itr.epoch, args.distributed_rank)), 'w+')
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if not args.cpu else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = generator.generate([model], sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                # Process top predictions
                for i, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if args.log_translations:
                        log.write('H-{}\t{}\t{}\n'.format(sample_id, hypo['score'], hypo_str))
                        log.write('P-{}\t{}\n'.format(
                             sample_id,
                             ' '.join(map(
                                 lambda x: '{:.4f}'.format(x),
                                 hypo['positional_scores'].tolist(),
                             ))
                        ))

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += sample['nsentences']

    if args.distributed_world_size > 1:
        _all_gather_bleu_scorer(scorer)
    if args.log_translations:
        log.close()
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    print('| Eval completed in: {:.2f}s'.format(time.time()-begin))

    return scorer.score(order=4)

def _all_gather_bleu_scorer(scorer):
    stats = distributed_utils.all_gather_list(scorer.stat)
    bleu_stat = bleu.BleuStat()
    bleu_stat.reflen  = reduce(lambda x,y: x+y, [s.reflen for s in stats])
    bleu_stat.predlen = reduce(lambda x,y: x+y, [s.predlen for s in stats])
    bleu_stat.match1  = reduce(lambda x,y: x+y, [s.match1 for s in stats])
    bleu_stat.count1  = reduce(lambda x,y: x+y, [s.count1 for s in stats])
    bleu_stat.match2  = reduce(lambda x,y: x+y, [s.match2 for s in stats])
    bleu_stat.count2  = reduce(lambda x,y: x+y, [s.count2 for s in stats])
    bleu_stat.match3  = reduce(lambda x,y: x+y, [s.match3 for s in stats])
    bleu_stat.count3  = reduce(lambda x,y: x+y, [s.count3 for s in stats])
    bleu_stat.match4  = reduce(lambda x,y: x+y, [s.match4 for s in stats])
    bleu_stat.count4  = reduce(lambda x,y: x+y, [s.count4 for s in stats])
    scorer.stat = bleu_stat


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best_loss'] = min(save_checkpoint.best, stats['loss'].avg)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint(\d+)\.pt')
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    else:
        print('| no existing checkpoint found {}'.format(checkpoint_path))
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
