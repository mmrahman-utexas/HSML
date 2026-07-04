from lib.train import train_a_task, test
from lib.consolidate import encode_weights
from lib.recall import recall
from lib.config import cfg
from lib.utils import log, compute_forgetting, confusion_matrix
from lib.dataset.mnist_variations import MNIST

import statistics
import os

import torch
import models.classifiers


def compute_recurring_metrics(result_val_t, result_val_a, eval_tasks):
    """Compute RA / LA / RecA / BTI for the recurring schedule.

    Mirrors compute_recurring_metrics.py:
      - RA  (retained):  average of the last snapshot row over the distinct tasks.
      - LA  (learned):   average accuracy of each task at its FIRST encounter.
      - RecA(recurring): average accuracy of each task at its SECOND encounter.
      - BTI:             average (end - first) over the distinct tasks.

    :param result_val_t: per-presentation original task id (the real train order).
    :param result_val_a: list of per-presentation accuracy vectors; columns are
                         aligned with `eval_tasks`.
    :param eval_tasks:   column labels (distinct task ids).
    :return: metrics dict, or None if any task is presented fewer than twice.
    """
    mat = [[float(v) for v in row] for row in result_val_a]
    n_rows = len(mat)
    col_of = {int(t): i for i, t in enumerate(eval_tasks)}
    pres = [int(t) for t in result_val_t]

    tasks, first_rows, second_rows = [], {}, {}
    for tid in pres:
        if tid not in first_rows:
            positions = [i for i, t in enumerate(pres) if t == tid]
            if len(positions) < 2:
                log('[RecurringMetrics] Task %d presented %d time(s) (<2); skipping.'
                    % (tid, len(positions)))
                return None
            tasks.append(tid)
            first_rows[tid] = positions[0]
            second_rows[tid] = positions[1]

    end_row = n_rows - 1

    def acc(row, tid):
        return mat[row][col_of[tid]]

    per_task = {}
    for tid in tasks:
        a1 = acc(first_rows[tid], tid)
        a2 = acc(second_rows[tid], tid)
        a_end = acc(end_row, tid)
        per_task[tid] = {
            'first_row': first_rows[tid],
            'second_row': second_rows[tid],
            'learned': a1,
            'recurring': a2,
            'end': a_end,
            'bti': a_end - a1,
        }

    n = len(tasks)
    return {
        'tasks_order': tasks,
        'retained_accuracy': sum(per_task[t]['end'] for t in tasks) / n,
        'learned_accuracy': sum(per_task[t]['learned'] for t in tasks) / n,
        'recurring_accuracy': sum(per_task[t]['recurring'] for t in tasks) / n,
        'bti': sum(per_task[t]['bti'] for t in tasks) / n,
        'per_task': per_task,
    }


def learn_continually():
    log('\nRunning experiments using MERLIN.')

    cfg.is_cifar_10 = 'cifar10' == cfg.continual.task
    cfg.is_cifar_100 = 'cifar100' == cfg.continual.task
    cfg.is_mini_imagenet = 'mini_imagenet' in cfg.continual.task

    recurring = cfg.continual.recurring.enable

    if recurring:
        # Recurring setting (mirrors La-MAML recurring_task_mode):
        # a small `subset` of distinct tasks is presented multiple times, each
        # presentation using one disjoint data-half of the task. `order` is the
        # schedule as [task_id, split_id] pairs. Eval / consolidation use `subset`.
        subset = [int(t) for t in cfg.continual.recurring.subset]
        n_splits = int(cfg.continual.recurring.n_splits)
        presentations = [(int(t), int(s)) for (t, s) in cfg.continual.recurring.order]
        eval_tasks = list(subset)
        load_tasks = list(subset)
        row_labels = [t for (t, _) in presentations]
        log('Recurring mode: subset=%s, n_splits=%d, presentations=%d'
            % (subset, n_splits, len(presentations)))
    else:
        tasks = list(range(cfg.continual.n_tasks))
        if cfg.continual.shuffle_task:
            tasks = torch.randperm(cfg.continual.n_tasks).tolist()
        eval_tasks = list(range(cfg.continual.n_tasks))
        presentations = [(t, None) for t in tasks]
        load_tasks = list(eval_tasks)
        row_labels = list(cfg.continual.task_order) if cfg.continual.task_order else list(tasks)

    observed_tasks = []
    final_accuracies = []
    individual_acc = []
    all_rows = []
    result_val_t = []   # task ID at each evaluation point  (mirrors train_versa_cosfan.py)
    result_val_a = []   # per-task accuracy vector at each evaluation point

    # Pre-load every training task and compute one sample permutation per task,
    # seeded by cfg.data_seed (analogous to La-MAML task_incremental_loader).
    # Test tasks are NOT permuted; permutations are only used to split train data
    # into disjoint subsets for each ensemble model. In recurring mode the
    # permutation for each task is further split into `n_splits` disjoint halves,
    # one per presentation of that task.
    log('Pre-loading training tasks and computing sample permutations (data_seed=%d).' % cfg.data_seed)
    perm_gen = torch.Generator()
    perm_gen.manual_seed(cfg.data_seed)
    all_train_datasets = {}
    sample_permutations = {}
    for t in load_tasks:
        ds = MNIST('./data', task=t, mode='Train', transform=None)
        perm = torch.randperm(len(ds), generator=perm_gen)
        all_train_datasets[t] = ds
        if recurring:
            # Disjoint contiguous halves; presentation s trains on split s.
            sample_permutations[t] = list(torch.chunk(perm, n_splits))
        else:
            sample_permutations[t] = perm

    baseline_model = getattr(models.classifiers, cfg.model)().to(cfg.device)
    baseline = [test(baseline_model, [task], verbose=False, mode='Test') for task in eval_tasks]
    all_rows.append(torch.tensor(baseline, dtype=torch.float32))

    for task, split in presentations:
        if split is None:
            log('\nLearning task %d.' % task)
        else:
            log('\nLearning task %d (split %d/%d).' % (task, split, n_splits))

        # Track distinct tasks observed so far (no duplicates on recurrence) so
        # consolidation / recall iterate over distinct tasks only.
        if task not in observed_tasks:
            observed_tasks.append(task)

        perm = sample_permutations[task][split] if recurring else sample_permutations[task]

        # Train multiple models for this presentation. Filenames are keyed by
        # task id, so a recurring presentation overwrites the previous models /
        # prior for that task with ones trained on fresh data.
        for model_id in range(cfg.n_models):
            train_a_task(task, model_id, all_train_datasets[task], perm)

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

    # In recurring mode rows (presentations) outnumber columns (distinct eval
    # tasks), so the default `post_rows[c, c]` diagonal is meaningless. Map each
    # eval-task column to the row of that task's LAST presentation, so the
    # learned / backward-transfer metrics are measured right after a task's final
    # exposure.
    diag_row_indices = None
    col_labels = row_labels
    if recurring:
        col_labels = eval_tasks
        diag_row_indices = []
        for eval_task in eval_tasks:
            last_row = max(i for i, (t, _) in enumerate(presentations) if t == eval_task)
            diag_row_indices.append(last_row)

    stats = confusion_matrix(
        result_a=result_a,
        log_dir=os.path.join(cfg.output_dir, 'logs'),
        fname='merlin_eval.txt',
        row_labels=row_labels,
        col_labels=col_labels,
        diag_row_indices=diag_row_indices,
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

    # Recurring-task metrics (RA / LA / RecA / BTI), matching compute_recurring_metrics.py.
    if recurring:
        rec_metrics = compute_recurring_metrics(result_val_t, result_val_a, eval_tasks)
        if rec_metrics is not None:
            log('\n####Recurring Metrics####')
            log('Tasks (first-appearance order): %s' % rec_metrics['tasks_order'])
            log('RA   (retained, last row avg):       %.6f' % rec_metrics['retained_accuracy'])
            log('LA   (learned, first encounter avg): %.6f' % rec_metrics['learned_accuracy'])
            log('RecA (recurring, second enc. avg):   %.6f' % rec_metrics['recurring_accuracy'])
            log('BTI  (end - first, avg):             %.6f' % rec_metrics['bti'])
            for tid in rec_metrics['tasks_order']:
                d = rec_metrics['per_task'][tid]
                log('  tid=%2d  first_row=%2d  second_row=%2d  learned=%.6f  recurring=%.6f  end=%.6f  bti=%+.6f'
                    % (tid, d['first_row'], d['second_row'], d['learned'],
                       d['recurring'], d['end'], d['bti']))
            torch.save(rec_metrics,
                       os.path.join(cfg.output_dir, 'pickles', 'recurring_metrics.pkl'))
