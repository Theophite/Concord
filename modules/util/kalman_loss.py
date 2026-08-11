"""Kalman loss meter: makes the diffusion training loss READABLE.

The raw per-step loss is dominated by which TIMESTEPS the batch drew -- E[loss|t]
varies several-fold across t, so batch-to-batch scatter is timestep roulette, not
model change, and an EMA's noise floor is set by that roulette. This meter:

  1. learns an online t-conditional BASELINE f(t) (32 buckets, normalized-LMS on the
     batch-mean observation y ~ sum_i f(t_i)/B -- unbiased for the linear model, no
     per-item losses needed), frozen after FREEZE_N updates so it stays a fixed yardstick;
  2. scores each step in the log domain, z = ln(y) - ln(x . f)   ("skill vs the
     t-matched expectation"), which removes the roulette;
  3. runs a 2-state Kalman filter [skill, trend] on z with adaptive measurement
     noise R (EW innovation variance, Myers-Tapley style), so the readout is
     "skill +3.4% vs baseline, trend +0.42 +/- 0.18 %/epoch" -- a drift estimate
     WITH a confidence interval instead of tea-leaves. (Sign: positive = BETTER.)

Degrades gracefully: with no timesteps available it filters ln(y) directly (still
trend + CI, just a higher noise floor). State round-trips through a json sidecar so
the exit-42 relaunch cadence doesn't reset it. Meter-only: never touches training.

Reading the line:
  [kloss] skill=+3.4% trend=+0.42+/-0.18%/ep (improving)   n=2314
  - skill: % BELOW the frozen early-run t-matched baseline. POSITIVE = better (loss below baseline).
  - trend: rate of skill CHANGE, %/epoch, with 2-sigma; POSITIVE = improving. "improving"/"flat"/
    2-sigma verdict. Teacher-forced caveat still applies: this reads DENOISING skill;
    generation quality needs samples.
"""
import json
import math
import os

N_BUCKETS = 32
T_MAX = 1000.0
WARMUP_N = 200          # global-stats-only window (no z emitted)
FREEZE_N = 1000         # baseline f(t) frozen after this many updates (~2 epochs @514)
NLMS_MU = 0.10          # baseline learning rate (normalized LMS)
Q_SKILL = (2.0e-4) ** 2  # per-update random-walk variance of skill (log domain)
Q_TREND = (1.0e-7) ** 2  # per-update random-walk variance of trend: real trends change over
                         # thousands of steps, and the CI scales ~ (Q_TREND*R)^(1/4) -- tight Q
                         # resolves ~0.5%/ep trends at the cost of lagging fast trend CHANGES
                         # by a few hundred steps (fine: trend turns are slow events)
R_MIN = 1e-4            # log-domain variance floor (1% std): adaptive R may NEVER collapse
                         # below this -- an R collapse sets the Kalman gain ~1 and the filter
                         # DIFFERENTIATES raw noise into the trend (observed: trend=-170%/ep)
R_INIT = 0.05 ** 2      # initial measurement noise (log domain)
VER = 2                 # state-format version (bump discards stale sidecars)


class KalmanLossMeter:
    def __init__(self, steps_per_epoch=514.0):
        self.spe = float(steps_per_epoch) or 514.0
        self.n = 0
        self.gmean = 0.0                       # global running mean of y (warmup + fallback)
        self.f = [0.0] * N_BUCKETS             # t-conditional baseline (loss units)
        self.f_seen = [0] * N_BUCKETS
        # KF state [skill, trend] in log domain + covariance
        self.s = 0.0
        self.t = 0.0
        self.P = [[1e-2, 0.0], [0.0, 1e-6]]
        self.R = R_INIT
        self.innov_ew = R_INIT                 # EW innovation^2, seeded at the prior -- NOT 0
                                               # (a zero seed underestimates R while warming and
                                               # collapses the filter; see R_MIN note)
        self.rvar = R_INIT                     # measured z-variance from late calibration:
        self.zmean = 0.0                       # the filter starts with MEASURED R and a seeded
        self.kf_started = False                # state, not guesses (a 12x-small R_INIT pinned
                                               # the gain at ~1 and the trend differentiated noise)
        self.s_ref = None                      # skill at baseline-freeze: cancels the Jensen
                                               # offset between loss-domain baseline and log-
                                               # domain scoring (a constant that would otherwise
                                               # read as phantom level); level = s - s_ref

    # ---------------- baseline ----------------
    def _buckets(self, timesteps):
        return [min(N_BUCKETS - 1, max(0, int(float(t) / T_MAX * N_BUCKETS)))
                for t in timesteps]

    def _baseline_update(self, y, bks):
        # y ~ sum_b x_b f_b with x_b = count_b / B: normalized LMS, unbiased for the
        # linear model even though only the batch MEAN is observed.
        if self.n > FREEZE_N:
            return
        B = float(len(bks))
        x = {}
        for b in bks:
            x[b] = x.get(b, 0.0) + 1.0 / B
        pred = sum(self.f[b] * w for b, w in x.items())
        xx = sum(w * w for w in x.values())
        err = y - pred
        for b, w in x.items():
            self.f[b] += NLMS_MU * err * w / max(xx, 1e-9)
            self.f_seen[b] += 1

    def _baseline_pred(self, bks):
        vals, miss = [], 0
        B = float(len(bks))
        pred = 0.0
        for b in bks:
            if self.f_seen[b] >= 5:
                pred += self.f[b] / B
            else:
                pred += self.gmean / B          # unseen bucket -> global fallback
                miss += 1
        return pred, miss

    # ---------------- filter ----------------
    def update(self, y, timesteps=None):
        """y = this update-step's (batch-mean) loss; timesteps = the batch's sampled
        timesteps (any iterable / 1-D tensor) or None. Returns the display string or
        None while warming up."""
        if not (y and math.isfinite(y) and y > 0):
            return self.line()
        self.n += 1
        self.gmean = self.gmean + (y - self.gmean) / min(self.n, 500)
        bks = self._buckets([float(v) for v in timesteps]) if timesteps is not None else None
        if bks:
            self._baseline_update(y, bks)
        # The KF starts only AFTER the baseline freezes: while NLMS is still chasing f(t),
        # z is nonstationary by construction and any trend read from it is fiction. The LATE
        # calibration window (baseline ~converged) doubles as the R measurement: the NLMS
        # residuals in the log domain ARE the measurement noise.
        if self.n <= FREEZE_N:
            if bks and self.n > FREEZE_N - 400:
                pred_c, _ = self._baseline_pred(bks)
                zc = math.log(y) - math.log(max(pred_c, 1e-12))
                a = 0.02
                self.zmean = (1 - a) * self.zmean + a * zc
                self.rvar = (1 - a) * self.rvar + a * (zc - self.zmean) ** 2
            return self.line()
        if not self.kf_started:                # KF start: measured R, state seeded at the
            self.kf_started = True             # calibration mean -> no convergence transient
            self.R = max(R_MIN, self.rvar)
            self.innov_ew = self.R
            self.s = self.zmean
            self.P = [[self.R, 0.0], [0.0, 1e-8]]
        if bks:
            pred, _ = self._baseline_pred(bks)
        else:
            pred = self.gmean
        z = math.log(y) - math.log(max(pred, 1e-12))
        # predict
        s_pr = self.s + self.t
        t_pr = self.t
        P = self.P
        P00 = P[0][0] + 2 * P[0][1] + P[1][1] + Q_SKILL
        P01 = P[0][1] + P[1][1]
        P11 = P[1][1] + Q_TREND
        # update (H = [1, 0])
        innov = z - s_pr
        S = P00 + self.R
        k0 = P00 / S
        k1 = P01 / S
        self.s = s_pr + k0 * innov
        self.t = t_pr + k1 * innov
        self.P = [[(1 - k0) * P00, (1 - k0) * P01],
                  [P01 - k1 * P00, P11 - k1 * P01]]
        # adaptive R: EW innovation^2, minus the state's own share (Myers-Tapley lite).
        # Floored at max(R_MIN, innov_ew/4): the state may absorb at most ~75% of the
        # observed innovation variance -- the gain can be wrong, never explosive.
        a = 0.01
        self.innov_ew = (1 - a) * self.innov_ew + a * (innov * innov)
        if self.n > FREEZE_N + 100:
            self.R = max(R_MIN, 0.25 * self.innov_ew, self.innov_ew - P00)
        if self.s_ref is None and self.n >= FREEZE_N + 200:
            self.s_ref = self.s                # freeze the level reference (offset cancels)
        return self.line()

    # ---------------- readout ----------------
    def line(self):
        if self.n <= WARMUP_N:
            return None
        if self.n <= FREEZE_N:
            return f"kloss calibrating {self.n}/{FREEZE_N + 200}"
        # SIGN CONVENTION: positive = BETTER. skill = % BELOW the frozen baseline; trend = %/epoch
        # of IMPROVEMENT. The raw filter state (self.s=log-loss level, self.t=its drift) stays
        # loss-domain (lower=better); we negate ONLY at display so the readout reads intuitively.
        # State + sidecar are unchanged -> no migration, safe to flip mid-run (meter is display-only).
        skill = (1.0 - math.exp(self.s - (self.s_ref or self.s))) * 100.0   # +% = below baseline = better
        if self.s_ref is None:
            return f"kloss warming {self.n}/{FREEZE_N + 200}"
        tr = -self.t * self.spe * 100.0                      # +%/epoch = skill RISING = improving
        sd = 2.0 * math.sqrt(max(self.P[1][1], 0.0)) * self.spe * 100.0
        verdict = ("improving" if tr - sd > 0 else
                   "regressing" if tr + sd < 0 else "flat")
        return f"skill={skill:+.1f}% trend={tr:+.2f}+-{sd:.2f}%/ep ({verdict})"

    # ---------------- persistence ----------------
    def to_dict(self):
        return {"ver": VER, "n": self.n, "gmean": self.gmean, "f": self.f, "f_seen": self.f_seen,
                "s": self.s, "t": self.t, "P": self.P, "R": self.R,
                "innov_ew": self.innov_ew, "spe": self.spe, "s_ref": self.s_ref,
                "kf_started": self.kf_started, "rvar": self.rvar, "zmean": self.zmean}

    @classmethod
    def from_dict(cls, d):
        m = cls(d.get("spe", 514.0))
        for k in ("n", "gmean", "s", "t", "R", "innov_ew"):
            setattr(m, k, d[k])
        m.f = list(d["f"]); m.f_seen = list(d["f_seen"]); m.P = [list(r) for r in d["P"]]
        m.s_ref = d.get("s_ref")
        # RESUME MUST NOT RE-SEED: kf_started/rvar/zmean were not persisted
        # originally, so every relaunch re-entered the KF-start branch --
        # zeroing the level state, discarding the adapted R, and collapsing
        # the trend covariance to 1e-8 (a confidently-wrong CI while the
        # trend absorbed the re-convergence transient; under the restart-
        # per-sample cadence the trend never stabilized). Migration for old
        # sidecars: a filter past FREEZE_N was by definition started, its
        # adapted R is the best rvar estimate, and its own level is the
        # best zmean -- the start branch then becomes a no-op if ever hit.
        m.kf_started = bool(d.get("kf_started", d.get("n", 0) > FREEZE_N))
        m.rvar = float(d.get("rvar", d.get("R", R_INIT)))
        m.zmean = float(d.get("zmean", d.get("s", 0.0)))
        return m

    def save(self, path):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh)
        except OSError:
            pass

    @classmethod
    def load(cls, path, steps_per_epoch=514.0):
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                if d.get("ver") == VER:
                    m = cls.from_dict(d)
                    # spe is a UNIT, not state: the trend is %/EPOCH, and
                    # epochs are measured in UPDATE steps, which change when
                    # gradient accumulation changes. Always trust the live
                    # trainer value over the sidecar's, or the trend and its
                    # CI are scaled by the stale ratio.
                    if steps_per_epoch and steps_per_epoch > 0:
                        m.spe = float(steps_per_epoch)
                    return m
        except (OSError, ValueError, KeyError):
            pass
        return cls(steps_per_epoch)
