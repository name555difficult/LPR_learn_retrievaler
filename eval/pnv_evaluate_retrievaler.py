import argparse
import os
import pickle

import numpy as np
import torch
import tqdm

from eval.pnv_evaluate import get_latent_vectors
from eval.utils import get_query_database_splits
from misc.utils import TrainingParams, set_seed
from models.model_factory import model_factory
from models.retrievaler import build_retrievaler, load_retrievaler_config


def evaluate(model, retrievaler, device, params: TrainingParams, retrievaler_cfg,
             log: bool = False, model_name: str = 'model',
             show_progress: bool = False):
    eval_database_files, eval_query_files = get_query_database_splits(params)
    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    ave_recall = []
    ave_one_percent_recall = []
    ave_mrr = []
    ave_survival = []
    ave_empty_far = []
    ave_pos_reject = []

    for database_file, query_file in zip(eval_database_files, eval_query_files):
        if 'CSWildPlaces' in params.dataset_name:
            location_name = database_file.split('_')[1]
            temp = query_file.split('_')[1]
        else:
            location_name = database_file.split('_')[0]
            temp = query_file.split('_')[0]
        assert location_name == temp, (
            'Database location: {} does not match query location: {}'.format(
                database_file, query_file
            )
        )

        with open(os.path.join(params.dataset_folder, database_file), 'rb') as f:
            database_sets = pickle.load(f)
        with open(os.path.join(params.dataset_folder, query_file), 'rb') as f:
            query_sets = pickle.load(f)

        temp_stats = evaluate_dataset(
            model, retrievaler, device, params, retrievaler_cfg,
            database_sets, query_sets, log=log, model_name=model_name,
            show_progress=show_progress,
        )
        stats[location_name] = temp_stats
        ave_one_percent_recall.append(temp_stats['ave_one_percent_recall'])
        ave_recall.append(temp_stats['ave_recall'])
        ave_mrr.append(temp_stats['ave_mrr'])
        ave_survival.append(temp_stats['survival_stage1'])
        ave_empty_far.append(temp_stats['empty_false_accept_rate'])
        ave_pos_reject.append(temp_stats['positive_reject_rate'])

    stats['average'] = {
        'ave_one_percent_recall': np.mean(ave_one_percent_recall),
        'ave_recall': np.mean(ave_recall, axis=0),
        'ave_mrr': np.mean(ave_mrr),
        'survival_stage1': np.mean(ave_survival),
        'empty_false_accept_rate': np.mean(ave_empty_far),
        'positive_reject_rate': np.mean(ave_pos_reject),
    }
    return stats


def evaluate_dataset(model, retrievaler, device, params: TrainingParams,
                     retrievaler_cfg, database_sets, query_sets,
                     log: bool = False, model_name: str = 'model',
                     show_progress: bool = False):
    recall = np.zeros(25)
    count = 0
    one_percent_recall = []
    mrr = []
    survival_stage1 = []
    empty_chunks = 0
    empty_false_accepts = 0
    positive_chunks = 0
    positive_rejects = 0

    database_embeddings = []
    query_embeddings = []

    model.eval()
    retrievaler.eval()

    for data_set in tqdm.tqdm(
        database_sets, disable=not show_progress, desc='Computing database embeddings'
    ):
        database_embeddings.append(get_latent_vectors(model, data_set, device, params))

    for data_set in tqdm.tqdm(
        query_sets, disable=not show_progress, desc='Computing query embeddings'
    ):
        query_embeddings.append(get_latent_vectors(model, data_set, device, params))

    for i in range(len(database_sets)):
        for j in range(len(query_sets)):
            if (
                (i == j and params.skip_same_run)
                or database_embeddings[i] is None
                or query_embeddings[j] is None
            ):
                continue
            if 'CSCampus3D' in params.dataset_name and i != 1:
                continue

            pair_recall, pair_opr, pair_mrr, pair_extra = get_recall_retrievaler(
                database_embeddings[i], query_embeddings[j],
                query_sets[j], database_sets[i], i,
                retrievaler, device, retrievaler_cfg,
                log=log, model_name=model_name, show_progress=show_progress,
            )
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            mrr.append(pair_mrr)
            survival_stage1.append(pair_extra['survival_stage1'])
            empty_chunks += pair_extra['empty_chunks']
            empty_false_accepts += pair_extra['empty_false_accepts']
            positive_chunks += pair_extra['positive_chunks']
            positive_rejects += pair_extra['positive_rejects']

    if count == 0:
        return {
            'ave_one_percent_recall': 0.0,
            'ave_recall': np.zeros(25),
            'ave_mrr': 0.0,
            'survival_stage1': 0.0,
            'empty_false_accept_rate': 0.0,
            'positive_reject_rate': 0.0,
        }

    ave_recall = recall / count
    empty_far = empty_false_accepts / empty_chunks if empty_chunks > 0 else 0.0
    pos_reject = positive_rejects / positive_chunks if positive_chunks > 0 else 0.0
    return {
        'ave_one_percent_recall': np.mean(one_percent_recall),
        'ave_recall': ave_recall,
        'ave_mrr': np.mean(mrr),
        'survival_stage1': np.mean(survival_stage1),
        'empty_false_accept_rate': empty_far,
        'positive_reject_rate': pos_reject,
    }


@torch.inference_mode()
def retrieve_query(query_vector, database_vectors, true_neighbors, retrievaler,
                   device, retrievaler_cfg, num_neighbors=25):
    if retrievaler_cfg.chunk_size <= 0:
        raise ValueError('retrievaler chunk_size must be positive')

    if torch.is_tensor(database_vectors):
        database = database_vectors.to(device=device, dtype=torch.float32)
    else:
        database = torch.as_tensor(
            np.asarray(database_vectors, dtype=np.float32), device=device
        )

    if torch.is_tensor(query_vector):
        query = query_vector.to(device=device, dtype=torch.float32).view(1, -1)
    else:
        query = torch.as_tensor(
            np.asarray(query_vector, dtype=np.float32), device=device
        ).view(1, -1)

    true_neighbors = set(int(e) for e in true_neighbors)
    database_size = database.shape[0]
    if database_size == 0:
        return [], {
            'survival_stage1': 0.0,
            'empty_chunks': 0,
            'empty_false_accepts': 0,
            'positive_chunks': 0,
            'positive_rejects': 0,
        }

    accepted = {}
    all_scores = []
    empty_chunks = 0
    empty_false_accepts = 0
    positive_chunks = 0
    positive_rejects = 0

    for start in range(0, database_size, retrievaler_cfg.chunk_size):
        end = min(start + retrievaler_cfg.chunk_size, database_size)
        chunk = database[start:end].unsqueeze(0)
        logits = retrievaler(query, chunk)
        scores = (logits[0, 1:] - logits[0, 0]).detach().cpu().numpy()
        chunk_indices = np.arange(start, end)

        max_delta = float(np.max(scores))
        true_positions = [
            pos for pos, db_idx in enumerate(chunk_indices)
            if int(db_idx) in true_neighbors
        ]
        if len(true_positions) > 0:
            positive_chunks += 1
            max_true_delta = float(np.max(scores[true_positions]))
            if max_true_delta <= retrievaler_cfg.tau:
                positive_rejects += 1
        else:
            empty_chunks += 1
            if max_delta > retrievaler_cfg.tau:
                empty_false_accepts += 1

        for pos, score in enumerate(scores):
            all_scores.append((int(chunk_indices[pos]), float(score)))

        top_m = min(retrievaler_cfg.top_m, len(scores))
        if top_m <= 0:
            continue
        top_pos = np.argpartition(-scores, kth=top_m - 1)[:top_m]
        top_pos = top_pos[np.argsort(-scores[top_pos])]
        for pos in top_pos:
            score = float(scores[pos])
            if score > retrievaler_cfg.tau:
                db_idx = int(chunk_indices[pos])
                if db_idx not in accepted or score > accepted[db_idx]:
                    accepted[db_idx] = score

    stage1_pool_ids = set(accepted.keys())
    survival = float(len(true_neighbors.intersection(stage1_pool_ids)) > 0)

    all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    pool = sorted(accepted.items(), key=lambda x: x[1], reverse=True)
    pool_ids = {idx for idx, _ in pool}
    min_pool = min(num_neighbors, database_size)
    if len(pool) < min_pool:
        for idx, score in all_scores:
            if idx in pool_ids:
                continue
            pool.append((idx, score))
            pool_ids.add(idx)
            if len(pool) >= min_pool:
                break

    if len(pool) == 0:
        return [], {
            'survival_stage1': survival,
            'empty_chunks': empty_chunks,
            'empty_false_accepts': empty_false_accepts,
            'positive_chunks': positive_chunks,
            'positive_rejects': positive_rejects,
        }

    stage2_limit = max(num_neighbors, retrievaler_cfg.stage2_max_candidates)
    pool = pool[:stage2_limit]
    pool_indices = np.array([idx for idx, _ in pool], dtype=np.int64)
    stage2_candidates = database[pool_indices].unsqueeze(0)
    final_logits = retrievaler(query, stage2_candidates)
    final_scores = (final_logits[0, 1:] - final_logits[0, 0]).detach().cpu().numpy()
    final_order = np.argsort(-final_scores)
    ranked = pool_indices[final_order][:num_neighbors].tolist()

    return ranked, {
        'survival_stage1': survival,
        'empty_chunks': empty_chunks,
        'empty_false_accepts': empty_false_accepts,
        'positive_chunks': positive_chunks,
        'positive_rejects': positive_rejects,
    }


def get_recall_retrievaler(database_output, queries_output, query_set,
                           database_set, database_index, retrievaler, device,
                           retrievaler_cfg, log=False, model_name='model',
                           show_progress=False):
    num_neighbors = 25
    recall = [0] * num_neighbors
    recall_idx = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    survival_stage1 = []
    empty_chunks = 0
    empty_false_accepts = 0
    positive_chunks = 0
    positive_rejects = 0
    if torch.is_tensor(database_output):
        database_tensor = database_output.to(device=device, dtype=torch.float32)
    else:
        database_tensor = torch.as_tensor(
            np.asarray(database_output, dtype=np.float32), device=device
        )

    for i in tqdm.tqdm(
        range(len(queries_output)), desc='Retrievaler ranking',
        leave=False, disable=not show_progress
    ):
        query_details = query_set[i]
        true_neighbors = query_details[database_index]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        ranked_indices, extra = retrieve_query(
            queries_output[i], database_tensor, true_neighbors, retrievaler,
            device, retrievaler_cfg, num_neighbors=num_neighbors
        )
        survival_stage1.append(extra['survival_stage1'])
        empty_chunks += extra['empty_chunks']
        empty_false_accepts += extra['empty_false_accepts']
        positive_chunks += extra['positive_chunks']
        positive_rejects += extra['positive_rejects']

        true_set = set(int(e) for e in true_neighbors)
        for j, db_idx in enumerate(ranked_indices):
            if db_idx in true_set:
                recall[j] += 1
                recall_idx.append(j + 1)
                break

        if len(set(ranked_indices[:threshold]).intersection(true_set)) > 0:
            one_percent_retrieved += 1

    if num_evaluated == 0:
        return [0.0] * num_neighbors, 0.0, 0.0, {
            'survival_stage1': 0.0,
            'empty_chunks': empty_chunks,
            'empty_false_accepts': empty_false_accepts,
            'positive_chunks': positive_chunks,
            'positive_rejects': positive_rejects,
        }

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    mrr = np.mean(1 / np.array(recall_idx)) * 100 if len(recall_idx) > 0 else 0.0
    return recall, one_percent_recall, mrr, {
        'survival_stage1': float(np.mean(survival_stage1)) if survival_stage1 else 0.0,
        'empty_chunks': empty_chunks,
        'empty_false_accepts': empty_false_accepts,
        'positive_chunks': positive_chunks,
        'positive_rejects': positive_rejects,
    }


def print_eval_stats_retrievaler(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = (
            'Avg. top 1% recall: {:.2f}   Avg. MRR: {:.2f}   '
            'Stage1 survival: {:.2f}   Empty FAR: {:.4f}   Pos reject: {:.4f}   '
            'Avg. recall @N:'
        )
        print(t.format(
            stats[database_name]['ave_one_percent_recall'],
            stats[database_name]['ave_mrr'],
            stats[database_name]['survival_stage1'],
            stats[database_name]['empty_false_accept_rate'],
            stats[database_name]['positive_reject_rate'],
        ))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats_retrievaler(file_name, prefix, stats):
    s = prefix
    with open(file_name, 'a') as f:
        for ds in stats:
            s += f"\n[{ds}]\n"
            s += (
                "AR@1%: {:0.2f}, AR@1: {:0.2f}, MRR: {:0.2f}, "
                "Stage1 survival: {:0.4f}, Empty FAR: {:0.4f}, "
                "Pos reject: {:0.4f}, AR@N:\n"
            ).format(
                stats[ds]['ave_one_percent_recall'],
                stats[ds]['ave_recall'][0],
                stats[ds]['ave_mrr'],
                stats[ds]['survival_stage1'],
                stats[ds]['empty_false_accept_rate'],
                stats[ds]['positive_reject_rate'],
            )
            s += str(stats[ds]['ave_recall'])
        s += "\n------------------------------------------------------------------------\n\n"
        f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate learned retrievaler')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Learned retrievaler checkpoint')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--log', dest='log', action='store_true',
                        help='Reserved for compatibility with pnv_evaluate.py')
    parser.set_defaults(log=False)

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Weights: {}'.format(args.weights))
    print('Debug mode: {}'.format(args.debug))

    set_seed()
    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()
    retrievaler_cfg = load_retrievaler_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = model_factory(params.model_params)
    retrievaler = build_retrievaler(params.model_params.output_dim, retrievaler_cfg)
    assert os.path.exists(args.weights), 'Cannot open checkpoint: {}'.format(args.weights)
    state = torch.load(args.weights, map_location=device)
    if 'model_state_dict' not in state or 'retrievaler_state_dict' not in state:
        raise KeyError('Expected a learned retrievaler checkpoint with model_state_dict and retrievaler_state_dict')
    model.load_state_dict(state['model_state_dict'])
    retrievaler.load_state_dict(state['retrievaler_state_dict'])
    model.to(device)
    retrievaler.to(device)

    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    model_name = os.path.split(args.weights)[1]
    prefix = "Model Params: {}, Config: {}, Model: {}".format(
        model_params_name, config_name, model_name
    )

    stats = evaluate(
        model, retrievaler, device, params, retrievaler_cfg,
        log=args.log, model_name=model_name, show_progress=True
    )
    print_eval_stats_retrievaler(stats)
    pnv_write_eval_stats_retrievaler(
        f"pnv_{params.dataset_name}_retrievaler_results.txt", prefix, stats
    )
