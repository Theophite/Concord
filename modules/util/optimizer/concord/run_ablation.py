"""Sequential CIFAR ablation runner for the Concord cf-discount.

Each (arm, seed) runs as a FRESH SUBPROCESS because the coherence on/off flags bake into Triton
constexprs at first kernel launch (can't flip cf mid-process). Runs one at a time (single GPU),
collects per-run JSON, prints a mean+/-std summary and the PAIRED cf-on minus cf-off delta on
the DEPLOY (consolidated) weight -- the load-bearing metric, since production drops s_fast.

What this measures:
  * STANDARD (dense, balanced) CIFAR arms = a correctness / stability / regression check, plus
    the cf knee sweep. NOT a benefit demo: dense CIFAR has little scattered residual for cf to
    protect, so expect B ~= A here. The point is "THIS version trains cleanly and the knobs
    behave", anchored to the external adamw_ref.
  * LONG-TAIL arms (--longtail 0.01) = the regime that actually stresses cf: rare-class
    gradients are the scattered/diverse residual cf is meant to protect. This is where a real
    cf benefit (if any) should show up, in deploy_acc and especially worst-class accuracy.
  * deploy_acc (consolidated weight, s_fast dropped) is the headline. live_acc is reported too.
  * Absolute accuracy is anchored to adamw_ref only; the packed-B net is NOT the core's
    validated Foliated net, so do not compare its absolute number to the core's CIFAR result.

LAUNCH ONLY WHEN THE GPU IS FREE. Examples:
    python run_ablation.py --smoke                       # 60 steps/arm, 1 seed -- shake out the harness
    python run_ablation.py --epochs 80 --seeds 0,1,2     # the real dense sweep with error bars
    python run_ablation.py --epochs 80 --seeds 0,1,2 --longtail 0.01   # + the cf-stress arms
    python run_ablation.py --epochs 80 --seeds 0,1,2 --ratio_coh 1 --sigmag_peak 0.6  # SDXL parity
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, 'train_cifar_cf.py')

# Each dict = the non-default flags for one arm. lam=0.025 keeps dimensionless dissipation at
# the production value; gf_consol=lam/lr is set in the driver. cf-on/off arms share lam so the
# ONLY difference is the cf-discount (set_coh_vhat).
STD_ARMS = [
    dict(arm='adamw_ref',    optimizer='adamw'),                                   # external reference
    dict(arm='A_cf_off',     optimizer='concord', coh_vhat=0, lam=0.025),          # control: dissipation on, cf off
    dict(arm='B_cf_on_k1',   optimizer='concord', coh_vhat=1, coh_kappa=1.0, lam=0.025),  # THE headline arm
    dict(arm='C_cf_k0p25',   optimizer='concord', coh_vhat=1, coh_kappa=0.25, lam=0.025), # knee sweep low
    dict(arm='C_cf_k4',      optimizer='concord', coh_vhat=1, coh_kappa=4.0, lam=0.025),  # knee sweep high
    dict(arm='D_no_gf_evap', optimizer='concord', coh_vhat=0, lam=0.0),            # no gf-evap (v_slow leak still on)
]


def lt_arms(factor):
    return [
        dict(arm='LT_A_cf_off',   optimizer='concord', coh_vhat=0, lam=0.025, longtail=factor),
        dict(arm='LT_B_cf_on_k1', optimizer='concord', coh_vhat=1, coh_kappa=1.0, lam=0.025, longtail=factor),
    ]


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]  # drop None/NaN
    if not xs:
        return float('nan'), float('nan'), 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, v ** 0.5, len(xs)


def paired_delta(by, arm_b, arm_a, key):
    """Per-seed (paired) arm_b - arm_a on `key`, aligned by seed order. Returns mean,std,n."""
    if arm_b not in by or arm_a not in by:
        return None
    a = {r['args']['seed']: r.get(key) for r in by[arm_a]}
    b = {r['args']['seed']: r.get(key) for r in by[arm_b]}
    deltas = [b[s] - a[s] for s in a.keys() & b.keys()
              if a[s] is not None and b[s] is not None]
    return mean_std(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--seeds', type=str, default='0,1,2', help='comma-separated seeds for error bars')
    ap.add_argument('--out_dir', type=str, default=os.path.join(HERE, 'cifar_ablation_results'))
    ap.add_argument('--only', type=str, default='', help='comma-separated arm names to run')
    ap.add_argument('--smoke', action='store_true', help='cap each run at 60 steps (harness shakeout)')
    ap.add_argument('--longtail', type=float, default=0.0, help='>0: also run LT_* arms at this imb factor (e.g. 0.01)')
    ap.add_argument('--sigmag_peak', type=float, default=0.0, help='>0 (e.g. 0.6): SDXL-parity grad noise on concord arms')
    ap.add_argument('--ratio_coh', type=int, default=0, help='1: SDXL-parity ratio-coh gate + disable_cohpre on concord arms')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip() != '']
    only = set(s.strip() for s in args.only.split(',') if s.strip()) if args.only else None
    arms = list(STD_ARMS) + (lt_arms(args.longtail) if args.longtail > 0.0 else [])

    by = {}
    for spec in arms:
        if only and spec['arm'] not in only:
            continue
        by.setdefault(spec['arm'], [])
        for seed in seeds:
            rj = os.path.join(args.out_dir, f"{spec['arm']}_seed{seed}.json")
            cmd = [sys.executable, DRIVER, '--epochs', str(args.epochs),
                   '--seed', str(seed), '--results_json', rj]
            if args.smoke:
                cmd += ['--max_steps', '60']
            if spec.get('optimizer', 'concord') == 'concord':
                if args.sigmag_peak > 0.0:
                    cmd += ['--sigmag_peak', str(args.sigmag_peak)]
                if args.ratio_coh:
                    cmd += ['--ratio_coh', '1', '--disable_cohpre', '1']
            for k, v in spec.items():
                cmd += [f'--{k}', str(v)]
            print('\n>>> ' + ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=False)
            if os.path.exists(rj):
                with open(rj) as f:
                    by[spec['arm']].append(json.load(f))
            else:
                print(f"!!! {spec['arm']} seed{seed} produced no results json (crashed?)", flush=True)

    # --- summary ---
    n_seed = len(seeds)
    print(f"\n=== CIFAR cf-discount ablation (epochs={args.epochs} seeds={seeds} "
          f"sigmag={args.sigmag_peak} ratio_coh={args.ratio_coh}) ===", flush=True)
    print(f"{'arm':<15}{'deploy_acc':>18}{'live_acc':>18}{'worst_cls':>14}{'nan':>6}", flush=True)
    for spec in arms:
        arm = spec['arm']
        runs = by.get(arm, [])
        if not runs:
            continue
        dm, ds, dn = mean_std([r.get('best_deploy_acc') for r in runs])
        lm, ls, _ = mean_std([r.get('best_live_acc') for r in runs])
        wm, ws, _ = mean_std([r.get('best_worst_class') for r in runs])
        anynan = any(r.get('nan') for r in runs)
        wc = f"{wm:.4f}+-{ws:.4f}" if wm == wm else "    -"
        print(f"{arm:<15}{dm:>10.4f}+-{ds:<5.4f}{lm:>10.4f}+-{ls:<5.4f}{wc:>14}{str(anynan):>6}", flush=True)

    # --- headline: paired cf-on minus cf-off on the DEPLOY weight ---
    print("\ncf-discount effect (paired per-seed, DEPLOY weight):", flush=True)
    for b_arm, a_arm, label in [('B_cf_on_k1', 'A_cf_off', 'dense  B-A deploy_acc'),
                                ('LT_B_cf_on_k1', 'LT_A_cf_off', 'longtl B-A deploy_acc')]:
        d = paired_delta(by, b_arm, a_arm, 'best_deploy_acc')
        if d:
            print(f"  {label}: {d[0]:+.4f} +- {d[1]:.4f}  (n={d[2]} seeds)", flush=True)
        w = paired_delta(by, b_arm, a_arm, 'best_worst_class')
        if w and w[2] and w[0] == w[0] and b_arm.startswith('LT'):
            print(f"  longtl B-A worst_class: {w[0]:+.4f} +- {w[1]:.4f}  (n={w[2]})", flush=True)
    print("\nNote: dense CIFAR is a regression/stability check (expect B~=A). The long-tail arms "
          "(--longtail 0.01) are where a real cf benefit, if any, should appear.", flush=True)


if __name__ == '__main__':
    main()
