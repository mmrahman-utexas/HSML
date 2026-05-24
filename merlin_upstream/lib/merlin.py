from lib.train import train_a_task, test
from lib.consolidate import encode_weights
from lib.recall import recall
from lib.config import cfg
from lib.utils import log, compute_forgetting, confusion_matrix

import statistics
import os

import torch
import models.classifiers


def learn_continually():
    log('\nRunning experiments using MERLIN.')

    cfg.is_cifar_10 = 'cifar10' == cfg.continual.task
    cfg.is_cifar_100 = 'cifar100' == cfg.continual.task
    cfg.is_mini_imagenet = 'mini_imagenet' in cfg.continual.task

    tasks = list(range(cfg.continual.n_tasks))
    if cfg.continual.shuffle_task:
        tasks = torch.randperm(cfg.continual.n_tasks).tolist()

    eval_tasks = list(range(cfg.continual.n_tasks))
    row_labels = list(cfg.continual.task_order) if cfg.continual.task_order else list(tasks)

    observed_tasks = []
    final_accuracies = []
    individual_acc = []
    all_rows = []
    result_val_t = []   # task ID at each evaluation point  (mirrors train_versa_cosfan.py)
    result_val_a = []   # per-task accuracy vector at each evaluation point

    baseline_model = getattr(models.classifiers, cfg.model)().to(cfg.device)
    baseline = [test(baseline_model, [task], verbose=False, mode='Test') for task in eval_tasks]
    all_rows.append(torch.tensor(baseline, dtype=torch.float32))

    for task in tasks:
        log('\nLearning task %d.' % task)
        observed_tasks.append(task)

        # Train multiple models for a task.
        for model_id in range(cfg.n_models):
            train_a_task(task, model_id)

        # Incrementally consolidate the model.
        encode_weights(task, observed_tasks)

        # Recall the model and compute accuracy on test set.
        acc, all_accuracies = recall(observed_tasks, eval_tasks=eval_tasks)
        final_accuracies.append(acc)
        individual_acc.append(all_accuracies)
        all_rows.append(torch.tensor(all_accuracies, dtype=torch.float32))
        result_val_t.append(task)
        result_val_a.append(all_accuracies)

    torch.save(final_accuracies, os.path.join(cfg.output_dir, 'pickles', 'merlin_final_accuracy.pkl'))
    torch.save(individual_acc, os.path.join(cfg.output_dir, 'pickles', 'merlin_all_accuracy.pkl'))
    result_a = torch.stack(all_rows, dim=0)
    stats = confusion_matrix(
        result_a=result_a,
        log_dir=os.path.join(cfg.output_dir, 'logs'),
        fname='merlin_eval.txt',
        row_labels=row_labels,
        col_labels=row_labels,
    )

    # Save in the same format as train_versa_cosfan.py:
    #   (result_t, result_a, stats)  stored to merlin_results.pt
    result_t_tensor = torch.tensor(result_val_t, dtype=torch.float32)
    result_a_tensor = torch.stack(
        [torch.tensor(a, dtype=torch.float32) for a in result_val_a], dim=0
    )
    torch.save(
        (result_t_tensor, result_a_tensor, stats),
        os.path.join(cfg.output_dir, 'pickles', 'merlin_results.pt'),
    )
    log('Confusion-matrix metrics: {}'.format([float('{:.4f}'.format(s)) for s in stats[:-1]]))
    log('Final Accuracy:')
    log(final_accuracies)

    log('Average Accuracy: ' + str(statistics.mean(final_accuracies)))
    log('Forgetting: ' + str(compute_forgetting(individual_acc)))
