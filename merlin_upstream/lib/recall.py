import models.classifiers
from lib.config import cfg
from lib.utils import log, Metrics, compute_accuracy, compute_offset
import lib
from models.chunked_vae import CHUNKED_VAE

import statistics
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lib.dataset.mnist_variations import MNIST
import torch.distributions as dist


def recall(observed_tasks, eval_tasks=None):
    if eval_tasks is None:
        eval_tasks = observed_tasks
    if cfg.recall.cumulative_prior:
        weights = get_weights_from_chunked_vae_cumulative_prior(observed_tasks)
        acc, all_accuracies = ensemble_and_evaluate_cumulative_prior(weights, observed_tasks, eval_tasks)
    else:
        weights = get_weights_from_chunked_vae(observed_tasks)
        acc, all_accuracies = ensemble_and_evaluate(weights, observed_tasks, eval_tasks)

    log('Test Accuracy: %f' % acc)
    return acc, all_accuracies


# ---------------------------------------------------------------------------
# Support / query split helpers
# ---------------------------------------------------------------------------

def _get_test_support_query_split(task):
    """Return (support_Subset, query_Subset) for a task's test data.

    The split is deterministic: seeded by (cfg.seed, task) so every call with
    the same arguments returns the same partition. The support set size is
    capped at half the test set to guarantee a non-empty query set.
    """
    test_data = MNIST('./data', task=task, mode='Test', transform=None)
    n_total = len(test_data)
    n_support = min(cfg.continual.validation_samples_per_task, n_total // 2)
    gen = torch.Generator()
    gen.manual_seed(cfg.data_seed * 10007 + task * 999983)
    perm = torch.randperm(n_total, generator=gen)
    support_ds = Subset(test_data, perm[:n_support].tolist())
    query_ds = Subset(test_data, perm[n_support:].tolist())
    return support_ds, query_ds


def _finetune_on_support(weight, support_ds):
    """Fine-tune a weight vector on a support dataset, return updated weight vector."""
    classification_model = getattr(models.classifiers, cfg.model)().to(cfg.device)
    new_state_dict = construct_state_dict_from_weights(classification_model, weight)
    classification_model.load_state_dict(new_state_dict)

    support_loader = DataLoader(support_ds, batch_size=cfg.batch_size_train, shuffle=True)
    opt = torch.optim.Adam(
        classification_model.parameters(),
        lr=cfg.continual.finetune_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    classification_model.train()
    for _ in range(cfg.continual.n_finetune_epochs):
        for x, y in support_loader:
            if 'FC' in cfg.model:
                x = x.view(-1, 28 * 28).to(cfg.device)
            else:
                x = x.to(cfg.device)
            y = y.to(cfg.device)
            loss = F.cross_entropy(classification_model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return torch.tensor(lib.train.kernels_to_vector(classification_model.state_dict()))


def _evaluate_on_query(weight, task, query_ds):
    """Load model from weight, evaluate on query_ds, return (accuracy, model)."""
    if cfg.is_cifar_10:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=10).to(cfg.device)
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=100).to(cfg.device)
    else:
        classification_model = getattr(models.classifiers, cfg.model)().to(cfg.device)

    new_state_dict = construct_state_dict_from_weights(classification_model, weight)
    classification_model.load_state_dict(new_state_dict)
    classification_model.eval()

    query_loader = DataLoader(
        query_ds, batch_size=cfg.batch_size_test, shuffle=cfg.continual.shuffle_datapoints
    )
    accuracy, _ = lib.train.evaluate_accuracy(classification_model, query_loader, task=task)
    return accuracy, classification_model


def _ensembled_prediction_on_query(task, clf_models, query_ds):
    """Ensemble prediction over query_ds for a given task."""
    query_loader = DataLoader(
        query_ds, batch_size=cfg.batch_size_test, shuffle=cfg.continual.shuffle_datapoints
    )
    accuracy_metric = Metrics()
    offset1, offset2 = compute_offset(task)

    for x, y in query_loader:
        if 'FC' in cfg.model:
            x = x.view(-1, 28 * 28).to(cfg.device)
        else:
            x = x.to(cfg.device)
        y = y.to(cfg.device)

        if cfg.is_cifar_100 or cfg.is_mini_imagenet:
            y_pred = torch.zeros((x.size(0), 100)).to(cfg.device)
        else:
            y_pred = torch.zeros((x.size(0), 10)).to(cfg.device)

        for model in clf_models:
            output = model(x)
            if cfg.is_cifar_100 or cfg.is_mini_imagenet:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:100].data.fill_(-10e10)
            elif cfg.is_cifar_10:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:10].data.fill_(-10e10)
            y_pred += output

        accuracy = compute_accuracy(y_pred, y)[0].item()
        accuracy_metric.update(accuracy)

    log('Accuracy of task %d is %f' % (task, accuracy_metric.avg))
    return accuracy_metric.avg


def ensemble_and_evaluate(weights, observed_tasks, eval_tasks=None):
    log('Analysing weights for ensembling.')
    if eval_tasks is None:
        eval_tasks = observed_tasks

    # Pre-compute deterministic support / query splits for every eval task.
    task_splits = {task: _get_test_support_query_split(task) for task in eval_tasks}

    classification_models = []
    for task, task_weight in enumerate(weights):
        classification_models_ensemble = []
        support_ds = task_splits.get(task, (None, None))[0]
        for weight in task_weight:
            # Fine-tune on the deterministic test-support set for this task.
            weight_ft = _finetune_on_support(weight, support_ds) if support_ds is not None \
                else torch.tensor(weight)
            query_ds = task_splits.get(task, (None, None))[1]
            acc, model = _evaluate_on_query(weight_ft, task, query_ds)
            if acc > cfg.kernels.ensembling.min_clf_accuracy:
                log('[Task: %d] Individual accuracies: %f' % (task, acc))
                classification_models_ensemble.append(model)
        classification_models.append(classification_models_ensemble)

    log('Ensembling results from %d tasks.' % len(classification_models))

    accuracies = []
    for task in eval_tasks:
        model_idx = task if task < len(classification_models) else len(classification_models) - 1
        query_ds = task_splits[task][1]
        acc = _ensembled_prediction_on_query(task, classification_models[model_idx], query_ds)
        accuracies.append(acc)

    acc = statistics.mean(accuracies)
    log('Average accuracy for ' + str(observed_tasks) + ' is ' + str(acc))
    return acc, accuracies


def ensembled_prediction_for_a_task(task, clf_models):
    test_data = MNIST('./data', task=task, mode='Test', transform=None)

    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size_test,
                                 shuffle=cfg.continual.shuffle_datapoints)

    accuracy_metric = Metrics()
    offset1, offset2 = compute_offset(task)

    for idx, (x, y) in enumerate(test_dataloader):
        if 'FC' in cfg.model:
            x = x.view(-1, 28 * 28).to(cfg.device)
        else:
            x = x.to(cfg.device)
        y = y.to(cfg.device)

        if cfg.is_cifar_100 or cfg.is_mini_imagenet:
            y_pred = torch.zeros((x.size()[0], 100)).to(cfg.device)
        else:
            y_pred = torch.zeros((x.size()[0], 10)).to(cfg.device)


        for model in clf_models:
            # Model Aggregation: Max Voting
            output = model(x)
            if cfg.is_cifar_100 or cfg.is_mini_imagenet:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:100].data.fill_(-10e10)
            elif cfg.is_cifar_10:
                output[:, :offset1].data.fill_(-10e10)
                output[:, offset2:10].data.fill_(-10e10)
            y_pred += output

        accuracy = compute_accuracy(y_pred, y)[0].item()
        accuracy_metric.update(accuracy)

    log('Accuracy of task %d is %f' % (task, accuracy_metric.avg))
    return accuracy_metric.avg


def evaluate_classification_model(weight, observed_tasks, verbose=True, mode='Test'):

    if cfg.is_cifar_10:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=10).to(cfg.device)
    elif cfg.is_cifar_100 or cfg.is_mini_imagenet:
        classification_model = getattr(lib.baselines.common, cfg.model)(nclasses=100).to(cfg.device)
    else:
        classification_model = getattr(models.classifiers, cfg.model)().to(cfg.device)

    if cfg.verbose:
        log(classification_model)

    new_state_dict = construct_state_dict_from_weights(classification_model, weight)
    classification_model.load_state_dict(new_state_dict)
    classification_model.eval()

    accuracy = lib.train.test(classification_model, observed_tasks, verbose=verbose, mode=mode)
    return accuracy, classification_model


def construct_state_dict_from_weights(model, weights):
    new_state_dict = {}
    index = 0

    for layer, weight in model.state_dict().items():
        offset = int(torch.prod(torch.tensor(weight.size())).item())
        new_state_dict[layer] = weights[index:index + offset].view(weight.size())
        index = index + offset

    return new_state_dict


def ensemble_and_evaluate_cumulative_prior(weights, observed_tasks, eval_tasks=None):
    log('Analysing weights for ensembling.')
    if eval_tasks is None:
        eval_tasks = observed_tasks

    # Pre-compute deterministic support / query splits for every eval task.
    task_splits = {task: _get_test_support_query_split(task) for task in eval_tasks}

    # Combine support sets from all eval tasks for cumulative fine-tuning.
    combined_support_ds = ConcatDataset([task_splits[t][0] for t in eval_tasks])

    classification_models = []
    for index, weight in enumerate(weights):
        weight_ft = _finetune_on_support(weight, combined_support_ds)
        # Use the first eval task as a representative for the threshold check.
        first_task = eval_tasks[0] if eval_tasks else 0
        acc, model = _evaluate_on_query(weight_ft, first_task, task_splits[first_task][1])
        if acc > cfg.kernels.ensembling.min_clf_accuracy:
            log('[%d / %d] Individual accuracies: %f' % (index, len(weights), acc))
            classification_models.append(model)

    log('Ensembling the models.')
    accuracies = []
    for task in eval_tasks:
        query_ds = task_splits[task][1]
        acc = _ensembled_prediction_on_query(task, classification_models, query_ds)
        accuracies.append(acc)

    acc = statistics.mean(accuracies)
    log('Average accuracy for ' + str(observed_tasks) + ' is ' + str(acc))
    return acc, accuracies


def get_weights_from_chunked_vae_cumulative_prior(observed_tasks):

    # Loading model
    model = CHUNKED_VAE(cfg.kernels.chunking.num_chunks).to(cfg.device)
    checkpoint = torch.load(cfg.recall.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    mean = []
    for task in observed_tasks:
        # Loading task-specific prior
        prior_location = cfg.output_dir + '/pickles/prior_' + str(task) + '.pkl'
        prior_mean, _ = torch.load(prior_location)
        mean.append(prior_mean.cpu().data.numpy()[0])

    prior_mean = torch.FloatTensor(np.average(mean, axis=0))
    prior_log_var = torch.ones_like(prior_mean)

    prior = dist.Normal(prior_mean, torch.sqrt(torch.exp(prior_log_var)))
    ensemble_weights = []
    for i in range(cfg.kernels.ensembling.max_num_of_models):
        z = prior.rsample().squeeze_().to(cfg.device)
        weight = lib.consolidate.decode_chunked_model(model, z).to('cpu')
        ensemble_weights.append(weight)

    return ensemble_weights


def get_weights_from_chunked_vae(observed_tasks):
    weights = []

    # Loading model
    model = CHUNKED_VAE(cfg.kernels.chunking.num_chunks).to(cfg.device)
    checkpoint = torch.load(cfg.recall.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for task in observed_tasks:
        # Loading task-specific prior
        prior_location = cfg.output_dir + '/pickles/prior_' + str(task) + '.pkl'
        prior_mean, prior_log_var = torch.load(prior_location)
        prior = dist.Normal(prior_mean, torch.sqrt(torch.exp(prior_log_var)))

        ensemble_weights = []
        for i in range(cfg.kernels.ensembling.max_num_of_models):
            z = prior.rsample().squeeze_().to(cfg.device)
            weight = lib.consolidate.decode_chunked_model(model, z).to('cpu')
            ensemble_weights.append(weight)

        weights.append(ensemble_weights)
    return weights
