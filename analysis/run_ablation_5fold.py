import os
import time
import pandas as pd

CKPT = 'results/tables/ablation_5fold_partial.csv'
FINAL = 'results/tables/ablation_5fold.csv'
SEED_FILE = 'results/tables/hybrid_loss_ablation.csv'   # fold 0, already run


def _done_set(df):
    """Runs already completed, keyed by (config, fold)."""
    if df is None or df.empty:
        return set()
    return {(str(r['config']), int(r['fold'])) for _, r in df.iterrows()}


def load_progress():
    """Resume from the partial file if present, otherwise seed from fold 0."""
    frames = []
    if os.path.exists(CKPT):
        frames.append(pd.read_csv(CKPT))
    elif os.path.exists(SEED_FILE):
        frames.append(pd.read_csv(SEED_FILE))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=['config', 'fold'], keep='last')


def save_progress(df):
    """Write atomically so an interrupted write cannot corrupt the file."""
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    tmp = CKPT + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, CKPT)


def run(configs=('ce', 'human_ls', 'cmp', 'hybrid'), folds=(0, 1, 2, 3, 4),
        n_epochs=5, batch_size=64, lam=0.5):
    from trainers.ablation_trainer import run_ablation_fold

    df = load_progress()
    done = _done_set(df)
    todo = [(c, f) for c in configs for f in folds if (c, f) not in done]

    print(f"completed: {len(done)}  remaining: {len(todo)}")
    if done:
        print("  already have:", sorted(done))
    if not todo:
        print("nothing to do")
    else:
        t0 = time.time()
        for i, (cfg, fold) in enumerate(todo, 1):
            r = run_ablation_fold(cfg, fold=fold, n_epochs=n_epochs,
                                  batch_size=batch_size, lam=lam)
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            df = df.drop_duplicates(subset=['config', 'fold'], keep='last')
            save_progress(df)          # persist after every single run
            el = (time.time() - t0) / 60
            eta = el / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] {cfg:9s} fold {fold}: "
                  f"acc={r['test_accuracy']:.4f} ece={r['ece_test']:.4f} "
                  f"slope={r['confidence_slope']:+.4f}   {el:.0f} min elapsed, ~{eta:.0f} min left")

    df = df.sort_values(['config', 'fold']).reset_index(drop=True)
    df.to_csv(FINAL, index=False)
    print(f"\nsaved {len(df)} runs to {FINAL}")
    return df


if __name__ == '__main__':
    run()
