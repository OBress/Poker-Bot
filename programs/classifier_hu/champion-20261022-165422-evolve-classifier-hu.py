"""Classifier candidate. Frozen template classifier_v1 (EVOLUTION.md §2).

Only the code between "# EVOLVE-BLOCK <name>" and "# END-EVOLVE-BLOCK" may change. The harness
creates one Classifier per session, feeds it every public event in order (observe) and asks for
a seat's style probabilities with posterior(seat, feats), which must return K non-negative finite
numbers (the harness normalizes them). PARAMS values are tuned within [low, high].
"""

import math

import numpy as np

from pokerbot.evolve.api import (
    BUCKETS,
    LINES,
    Event,
    HandEnd,
    HandStart,
    PlayerAction,
    SeatFeatures,
    Showdown,
    StyleProfiles,
    param,
)

# EVOLVE-BLOCK params
PARAMS = {
    "w_bb": param(0.4239642091825956, 0.0, 2.0),
    "w_gauss": param(0.8107600729450106, 0.0, 2.0),
    "w_sd": param(0.20093849233178177, 0.0, 2.0),
    "tau": param(2.1820574777026116, 0.3, 4.0),
    "prior_power": param(1.7490933036050438, 0.0, 3.0),
    "shrink": param(0.08812647956706622, 0.0, 0.95),
    "ridge": param(0.10680886034360602, 1e-4, 1.0),
    "s2_floor": param(0.17155392531360236, 1e-6, 0.5),
    "imp_power": param(0.4000610230924709, 0.0, 2.0),
    "boost_pos": param(0.5909136655501186, 0.5, 2.5),
    "boost_street": param(1.322447835659609, 0.5, 2.5),
    "boost_action": param(1.3133411489331297, 0.5, 3.0),
    "w_size": param(-0.5132649430793854, -1.5, 1.5),
    "size_center": param(0.195731500236316, 0.0, 7.0),
    "size_shrink": param(6.686768224880671, 0.1, 50.0),
    "w_shallow": param(2.105236185591511, 0.0, 2.5),
    "shallow_center": param(0.6649023870090983, 0.0, 1.0),
    "shallow_shrink": param(4.748482322649321, 0.0, 50.0),
    "rare_mix": param(0.9090626228504616, 0.0, 1.0),
    "w_ep": param(-0.6959084332314547, -1.0, 1.0),
    "ep_center": param(0.23776637817417118, 0.0, 0.6),
    "ep_shrink": param(17.98341831115029, 0.0, 50.0),
    "fam_red": param(0.23710868426058251, 0.0, 1.0),
    "fam_corr": param(0.45, 0.0, 1.0),
    "w_dep": param(1.2612366326684905, 0.0, 3.0),
    "dep_center": param(57.56157197430232, 10.0, 150.0),
    "dep_width": param(11.785741352575027, 5.0, 100.0),
    "w_limp": param(0.40515144662753916, 0.0, 2.0),
    "limp_center": param(0.04851578972261303, 0.0, 0.5),
    "limp_shrink": param(3.594329913521432, 0.0, 50.0),
    "corr_lam": param(0.31509780393737014, 0.0, 1.0),
    "corr_floor": param(0.7808975673481094, 0.0, 0.95),
    "corr_power": param(1.0016713262664032, 0.5, 2.0),
    "corr_within": param(0.6732543891395748, 0.0, 1.0),
    "corr_within_floor": param(0.21556160444151598, 0.0, 0.9),
    "corr_global": param(0.6000571129087866, 0.0, 2.0),
    "obs_corr": param(0.2534229871526164, 0.0, 0.95),
    "w_ratio": param(0.15955461993287143, 0.0, 2.0),
    "ratio_shrink": param(29.385809738762745, 0.0, 100.0),
    "w_posgrad": param(0.31619652166757284, 0.0, 2.0),
    "posgrad_shrink": param(5.266944755335797, 0.0, 100.0),
    "w_c_raise": param(0.289312539099782, 0.0, 3.0),
    "w_c_limp": param(1.8116974070812129, 0.0, 3.0),
    "w_c_pos": param(0.7998392560273117, 0.0, 3.0),
    "comp_kappa": param(9.836495705923697, 1.0, 80.0),
    "w_posshape": param(0.15993732488470558, 0.0, 1.5),
    "posshape_noise": param(0.47753840716826856, 0.05, 3.0),
    "w_depshape": param(0.5260292989142119, 0.0, 1.5),
    "depshape_noise": param(0.3869320982398606, 0.05, 3.0),
    "w_bshape": param(0.15874138410320882, 0.0, 1.5),
    "bshape_kappa": param(2.7211738581809723, 0.5, 50.0),
    "w_sdmean": param(0.15077883494692826, 0.0, 1.5),
    "sdmean_noise": param(0.7388850793588084, 0.2, 4.0),
    "pool_mix": param(0.18962892369611828, 0.0, 1.0),
}
# END-EVOLVE-BLOCK

# EVOLVE-BLOCK helpers
def _lgamma(x):
    x = np.asarray(x, dtype=float)
    small = x < 0.5
    xr = np.where(small, 1.0 - x, x) - 1.0
    a = (0.99999999999980993
         + 676.5203681218851 / (xr + 1.0)
         - 1259.1392167224028 / (xr + 2.0)
         + 771.32342877765313 / (xr + 3.0)
         - 176.61502916214059 / (xr + 4.0)
         + 12.507343278686905 / (xr + 5.0)
         - 0.13857109526572012 / (xr + 6.0)
         + 9.9843695780195716e-6 / (xr + 7.0)
         + 1.5056327351493116e-7 / (xr + 8.0))
    t = xr + 7.5
    res = 0.5 * np.log(2.0 * np.pi) + (xr + 0.5) * np.log(t) - t + np.log(a)
    x_safe = np.where(small, x, 0.5)
    refl = np.log(np.pi) - np.log(np.sin(np.pi * x_safe)) - res
    return np.where(small, refl, res)


def _log_beta(a, b):
    return _lgamma(a) + _lgamma(b) - _lgamma(a + b)


def _softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _logit_rate(x, n):
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    p = (x + 0.5) / (n + 1.0)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))
# END-EVOLVE-BLOCK


class Classifier:
    def __init__(self, profiles: StyleProfiles, params: dict[str, float]) -> None:
        self.profiles = profiles
        self.p = params
        self.reset()

    def reset(self) -> None:
        # EVOLVE-BLOCK state
        self.beta_a = self.profiles.beta_a
        self.beta_b = self.profiles.beta_b
        self.log_beta_ab = _log_beta(self.beta_a, self.beta_b)
        self.means = self.profiles.means
        self.covs = self.profiles.covs
        self.log_shares = np.log(self.profiles.shares)
        self.K = self.means.shape[0]
        self.D = self.means.shape[1]
        self.I = np.eye(self.D)
        self.hands_idx = [i for i, f in enumerate(self.profiles.features) if f == "hands"]
        self.init_depth = {}

        mu = np.asarray(self.means, dtype=float)
        var_between = mu.var(axis=0)
        var_within = np.zeros(self.D)
        for k in range(self.K):
            var_within += np.diag(np.asarray(self.covs[k], dtype=float))
        var_within = var_within / max(self.K, 1)
        ratio = var_between / (var_within + 1e-6)
        ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), ratio, 0.0)
        nz = ratio > 0.0
        if nz.any():
            ratio = ratio / ratio[nz].mean()
        ratio = np.minimum(ratio, 10.0)

        power = float(self.p["imp_power"])
        w = np.power(np.maximum(ratio, 1e-6), power)

        pos_kw = ("vpip_", "pfr_", "rfi_")
        street_kw = ("agg_", "cbet", "donk", "xr", "betsize")
        action_kw = ("limp", "steal", "3bet", "4bet")
        bp = float(self.p["boost_pos"])
        bs = float(self.p["boost_street"])
        ba = float(self.p["boost_action"])
        names = list(self.profiles.features)
        for i, name in enumerate(names):
            mult = 1.0
            for kw in pos_kw:
                if kw in name:
                    mult *= bp
                    break
            for kw in street_kw:
                if kw in name:
                    mult *= bs
                    break
            for kw in action_kw:
                if kw in name:
                    mult *= ba
                    break
            w[i] = w[i] * mult

        for i in self.hands_idx:
            w[i] = 0.0
        self.feat_w = w

        agg_idx = []
        for i, name in enumerate(names):
            if "fold" in name:
                continue
            sign = 0.0
            if (("agg_" in name) or ("cbet" in name) or ("donk_" in name)
                    or ("xr_" in name) or ("3bet" in name) or ("4bet" in name)
                    or ("steal" in name) or (name == "pfr") or ("pfr_" in name)):
                sign = 1.0
            if "limp" in name:
                sign = -1.0
            if sign != 0.0:
                agg_idx.append((i, sign))
        aggr = np.zeros(self.K)
        if agg_idx:
            for i, sign in agg_idx:
                v = mu[:, i]
                sdv = v.std()
                if sdv > 1e-9:
                    aggr += sign * (v - v.mean()) / sdv
        else:
            for i, name in enumerate(names):
                if "vpip" in name:
                    v = mu[:, i]
                    sdv = v.std()
                    if sdv > 1e-9:
                        aggr += (v - v.mean()) / sdv
                    break
        sdv = aggr.std()
        if sdv > 1e-9:
            aggr = (aggr - aggr.mean()) / sdv
        else:
            aggr = np.zeros(self.K)
        self.style_aggr = aggr

        tight = np.zeros(self.K)
        for i, name in enumerate(names):
            if name == "vpip":
                v = mu[:, i]
                sdv = v.std()
                if sdv > 1e-9:
                    tight = -(v - v.mean()) / sdv
                break
        sdv = tight.std()
        if sdv > 1e-9:
            tight = (tight - tight.mean()) / sdv
        self.style_tight = tight

        limp_i = -1
        for i, name in enumerate(names):
            if name == "limp":
                limp_i = i
                break
        style_limp = np.zeros(self.K)
        if limp_i >= 0:
            v = mu[:, limp_i]
            sdv = v.std()
            if sdv > 1e-9:
                style_limp = (v - v.mean()) / sdv
        self.style_limp = style_limp

        self.rare_idx = int(np.argmin(self.profiles.shares))
        rare = np.zeros(self.K)
        rare[self.rare_idx] = 1.0
        depth_idx = []
        for i, name in enumerate(names):
            if ("_d0" in name or "_d1" in name) and ("vpip" in name or "pfr" in name):
                depth_idx.append(i)
        depth_aff = np.zeros(self.K)
        if depth_idx:
            vals = mu[:, depth_idx].mean(axis=1)
            sdv = vals.std()
            if sdv > 1e-9:
                depth_aff = (vals - vals.mean()) / sdv
        mix = float(self.p["rare_mix"])
        self.short_vec = mix * rare + (1.0 - mix) * depth_aff
        sdv = self.short_vec.std()
        if sdv > 1e-9:
            self.short_vec = (self.short_vec - self.short_vec.mean()) / sdv

        self.families = []
        fam_map = {}
        for i, name in enumerate(names):
            if name == "vpip" or name.startswith("vpip_"):
                key = "vpip"
            elif name == "pfr" or name.startswith("pfr_"):
                key = "pfr"
            elif name == "rfi" or name.startswith("rfi_"):
                key = "rfi"
            elif (name.startswith("cbet_") or name.startswith("agg_")
                  or name.startswith("donk_") or name.startswith("xr_")):
                st = "x"
                if "flop" in name:
                    st = "flop"
                elif "turn" in name:
                    st = "turn"
                elif "river" in name:
                    st = "river"
                key = "street_aggr_" + st
            elif name.startswith("fold_to_cbet_"):
                st = "x"
                if "flop" in name:
                    st = "flop"
                elif "turn" in name:
                    st = "turn"
                elif "river" in name:
                    st = "river"
                key = "fold_cbet_" + st
            elif name.startswith("betsize_"):
                key = "betsize"
            else:
                key = "uniq_" + name
            if key not in fam_map:
                fam_map[key] = []
                self.families.append(fam_map[key])
            fam_map[key].append(i)

        # Data-driven redundancy map: features whose style-mean vectors point in the
        # same direction are measuring the same trait, so their Beta-Binomial evidence
        # should not be counted independently.
        fam_id = np.full(self.D, -1, dtype=int)
        for f, fam in enumerate(self.families):
            for i in fam:
                fam_id[i] = f
        mu2 = np.asarray(self.means, dtype=float)
        mu2 = np.where(np.isfinite(mu2), mu2, 0.0)
        mc = mu2 - mu2.mean(axis=0, keepdims=True)
        norm = np.sqrt((mc * mc).sum(axis=0))
        norm = np.where(norm > 1e-9, norm, 1.0)
        mn = mc / norm
        sim = np.minimum(np.abs(mn.T @ mn), 1.0)
        same = fam_id[:, None] == fam_id[None, :]
        sim = np.where(same, 0.0, sim)
        floor = float(self.p["corr_floor"])
        if floor < 1.0:
            sim = np.where(sim >= floor, (sim - floor) / (1.0 - floor), 0.0)
        else:
            sim = np.zeros_like(sim)
        sim = np.power(np.maximum(sim, 0.0), float(self.p["corr_power"]))

        # Siblings inside one family (vpip_EP / vpip_MP / vpip_BTN, vpip_d0 / vpip_d1,
        # agg_flop / cbet_flop) are just as redundant as cross-family pairs, so feed the
        # same smooth correlation map over all pairs instead of relying only on the
        # crude "keep the largest weight in the family" rule.
        fc = float(self.p["fam_corr"])
        if fc > 0.0:
            s_all = np.minimum(np.abs(mn.T @ mn), 1.0)
            if floor < 1.0:
                s_all = np.where(s_all >= floor, (s_all - floor) / (1.0 - floor), 0.0)
            else:
                s_all = np.zeros_like(s_all)
            s_all = np.power(np.maximum(s_all, 0.0), float(self.p["corr_power"]))
            s_all = s_all * (1.0 - self.I)
            sim = np.maximum(sim, fc * s_all)
        self.corr_sim = sim

        # Contrast axes: after removing redundant evidence, add back the orthogonal
        # differences between correlated stats (PFR vs VPIP, late vs early position).
        self.name_to_i = {name: i for i, name in enumerate(names)}
        self.f_mean = np.asarray(self.profiles.f_mean, dtype=float)
        self.f_std = np.asarray(self.profiles.f_std, dtype=float)
        self.f_std = np.where(np.isfinite(self.f_std) & (self.f_std > 1e-9), self.f_std, 1.0)

        i_v = self.name_to_i.get("vpip", -1)
        i_p = self.name_to_i.get("pfr", -1)
        self.i_vpip = i_v
        self.i_pfr = i_p
        ratio_style = np.zeros(self.K)
        self.ratio_offset = 0.0
        self.ratio_scale = 1.0
        if i_v >= 0 and i_p >= 0:
            raw = mu2[:, i_p] - mu2[:, i_v]
            raw = np.where(np.isfinite(raw), raw, 0.0)
            self.ratio_offset = raw.mean()
            sd = raw.std()
            if sd > 1e-9:
                self.ratio_scale = sd
                ratio_style = (raw - self.ratio_offset) / sd
        self.ratio_style = ratio_style

        pos_style = np.zeros(self.K)
        self.pos_offset = 0.0
        self.pos_scale = 1.0
        self.pos_pairs = []
        for pos in ("EP", "MP", "CO", "BTN"):
            vn = "vpip_" + pos
            pn = "pfr_" + pos
            if vn in self.name_to_i and pn in self.name_to_i:
                self.pos_pairs.append((pos, vn, pn, self.name_to_i[vn], self.name_to_i[pn]))
        if self.pos_pairs:
            early = []
            late = []
            for pos, vn, pn, vi, pi in self.pos_pairs:
                vec = 0.5 * (mu2[:, vi] + mu2[:, pi])
                if pos in ("EP", "MP"):
                    early.append(vec)
                else:
                    late.append(vec)
            if early and late:
                raw = (np.mean(np.stack(late, axis=1), axis=1)
                       - np.mean(np.stack(early, axis=1), axis=1))
                raw = np.where(np.isfinite(raw), raw, 0.0)
                self.pos_offset = raw.mean()
                sd = raw.std()
                if sd > 1e-9:
                    self.pos_scale = sd
                    pos_style = (raw - self.pos_offset) / sd
        self.pos_style = pos_style
        # END-EVOLVE-BLOCK

    def observe(self, event: Event) -> None:
        # EVOLVE-BLOCK observe
        m = _obs_mod = None
        try:
            from_event = event
        except Exception:
            return
        if not isinstance(event, HandStart):
            return
        try:
            st = event.stacks
            bb = float(event.big_blind)
        except Exception:
            return
        if not (bb > 0.0):
            return
        try:
            n = len(st)
        except Exception:
            return
        for i in range(n):
            try:
                v = float(st[i])
            except Exception:
                continue
            if v > 0.0 and i not in self.init_depth:
                self.init_depth[i] = v / bb
        # END-EVOLVE-BLOCK

    def posterior(self, seat: int, feats: SeatFeatures) -> np.ndarray:
        # EVOLVE-BLOCK posterior
        prof = self.profiles
        p = self.p
        n = np.array([feats.n(f) for f in prof.features], dtype=float)
        x = np.array([feats.x(f) for f in prof.features], dtype=float)

        for idx in self.hands_idx:
            n[idx] = 0.0
            x[idx] = 0.0

        D = self.D
        K = self.K
        eyeD = np.eye(D)
        within = np.zeros((D, D))
        for k in range(K):
            Ck = np.asarray(self.covs[k], dtype=float)
            dg = np.diag(Ck).copy()
            dg = np.where(np.isfinite(dg) & (dg > 1e-9), dg, 1.0)
            inv = 1.0 / np.sqrt(dg)
            Rk = Ck * inv[:, None] * inv[None, :]
            within += np.where(np.isfinite(Rk), Rk, 0.0)
        within = within / max(K, 1)
        within = np.clip(within, -0.95, 0.95)
        within = 0.5 * (within + within.T)
        within = within * (1.0 - eyeD) + eyeD
        within_abs = np.abs(within) * (1.0 - eyeD)

        a = self.beta_a
        b = self.beta_b
        N = n[None, :]
        X = x[None, :]
        bb_terms = (_lgamma(X + a) + _lgamma(N - X + b) - _lgamma(N + a + b)
                    - self.log_beta_ab)

        active = n > 0.0
        red = float(p["fam_red"])
        mult = np.ones(self.D)
        if red > 0.0:
            for fam in self.families:
                idx = []
                for i in fam:
                    if active[i]:
                        idx.append(i)
                if len(idx) > 1:
                    s = 0.0
                    mx = 0.0
                    for i in idx:
                        wv = self.feat_w[i]
                        s += wv
                        if wv > mx:
                            mx = wv
                    if s > 0.0:
                        factor = (1.0 - red) + red * (mx / s)
                        for i in idx:
                            mult[i] = factor
        wa = np.where(active, self.feat_w * mult, 0.0)

        lam = float(p["corr_lam"])
        cw = float(p["corr_within"])
        if cw > 0.0:
            wf = float(p["corr_within_floor"])
            wf = min(max(wf, 0.0), 0.95)
            if wf < 1.0:
                wsim = np.where(within_abs >= wf, (within_abs - wf) / (1.0 - wf), 0.0)
            else:
                wsim = np.zeros_like(within_abs)
            wsim = np.power(np.maximum(wsim, 0.0), float(p["corr_power"]))
            sim = np.maximum(self.corr_sim, cw * wsim)
        else:
            sim = self.corr_sim
        if lam > 0.0:
            redund = sim @ wa
            denom = wa + lam * redund + 1e-12
            wa = np.where(wa > 0.0, (wa * wa) / denom, 0.0)

        cg = float(p["corr_global"])
        if cg > 0.0:
            am = (wa > 0.0).astype(float)
            cnt = max(float(am.sum()) - 1.0, 1.0)
            overlap = (within_abs @ am) / cnt
            wa = wa / (1.0 + cg * overlap)

        if (wa > 0.0).any():
            den = wa[active].mean()
            if (not np.isfinite(den)) or den <= 0.0:
                den = 1.0
            wa = wa / den
        else:
            wa = np.where(active, 1.0, 0.0)
        ll_bb = (bb_terms * wa[None, :]).sum(axis=1)

        z, s2 = prof.transform(n, x)
        z = np.asarray(z, dtype=float)
        s2 = np.asarray(s2, dtype=float)
        z = np.where(np.isfinite(z), z, 0.0)
        s2 = np.where(np.isfinite(s2) & (s2 >= 0.0), s2, 1e6)
        mask = n > 0.0
        z = np.where(mask, z, 0.0)
        s2 = np.where(mask, s2, 1e6)
        s2 = np.maximum(s2, p["s2_floor"])

        eye = self.I
        shrink = p["shrink"]
        ridge = p["ridge"]
        oc = float(p["obs_corr"])
        active_f = (n > 0.0).astype(float)
        if oc > 0.0:
            R_obs = eyeD + (within - eyeD) * active_f[:, None] * active_f[None, :]
            sdv = np.sqrt(np.maximum(s2, 0.0))
            corr_cov = sdv[:, None] * R_obs * sdv[None, :]
            obs_cov = (1.0 - oc) * np.diag(s2) + oc * corr_cov
            obs_cov = 0.5 * (obs_cov + obs_cov.T)
        else:
            obs_cov = np.diag(s2)
        ll_g = np.empty(self.K)
        for k in range(self.K):
            cov = (1.0 - shrink) * self.covs[k] + shrink * eye
            cov = cov + obs_cov + ridge * eye
            cov = 0.5 * (cov + cov.T)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov + 1e-3 * eye)
            diff = z - self.means[k]
            sol = np.linalg.solve(L, diff)
            mahal = sol @ sol
            logdet = 2.0 * np.log(np.diag(L)).sum()
            ll_g[k] = -0.5 * (mahal + logdet)

        sd = (np.log(prof.showdown + 1e-12) * feats.shown).sum(axis=(1, 2))

        size_names = ("betsize_b0", "betsize_b1", "betsize_b2", "betsize_b3",
                      "betsize_b4", "betsize_b5", "betsize_b6", "betsize_b7")
        tot = 0.0
        acc = 0.0
        for i, nm in enumerate(size_names):
            xi = feats.x(nm)
            if xi > 0.0:
                tot += xi
                acc += i * xi
        if tot > 0.0:
            avg_size = acc / tot
            conf = tot / (tot + p["size_shrink"] + 1e-9)
            sz = (avg_size - p["size_center"]) / 3.5
            sz = np.clip(sz, -1.5, 1.5) * conf
            aux = p["w_size"] * self.style_aggr * sz
        else:
            aux = 0.0

        h = max(float(feats.hands), 1.0)
        shallow = feats.n("vpip_d0") + feats.n("vpip_d1")
        frac = np.clip(shallow / h, 0.0, 1.0)
        conf = h / (h + p["shallow_shrink"] + 1e-9)
        sh = (frac - p["shallow_center"]) * conf
        aux = aux + p["w_shallow"] * self.short_vec * sh

        ep_n = feats.n("pfr_EP")
        if ep_n > 0.0:
            ep_rate = feats.x("pfr_EP") / ep_n
            conf = ep_n / (ep_n + p["ep_shrink"] + 1e-9)
            ep = (ep_rate - p["ep_center"]) * conf
            aux = aux + p["w_ep"] * self.style_aggr * ep

        dep = self.init_depth.get(seat, None)
        if dep is not None and dep > 0.0:
            wid = float(p["dep_width"])
            if wid < 1e-3:
                wid = 1e-3
            arg = (dep - p["dep_center"]) / wid
            arg = max(-60.0, min(60.0, arg))
            sc = 1.0 / (1.0 + math.exp(arg))
            aux = aux + p["w_dep"] * self.short_vec * sc

        nl = feats.n("limp")
        if nl > 0.0:
            lconf = nl / (nl + p["limp_shrink"] + 1e-9)
            lsc = (feats.x("limp") / nl - p["limp_center"]) * lconf
            aux = aux + p["w_limp"] * self.style_limp * lsc

        iv = self.i_vpip
        ip = self.i_pfr
        if iv >= 0 and ip >= 0:
            nv = feats.n("vpip")
            npf = feats.n("pfr")
            if nv > 0.0 and npf > 0.0:
                zv = (_logit_rate(feats.x("vpip"), nv) - self.f_mean[iv]) / self.f_std[iv]
                zp = (_logit_rate(feats.x("pfr"), npf) - self.f_mean[ip]) / self.f_std[ip]
                obs = (zp - zv - self.ratio_offset) / (self.ratio_scale + 1e-9)
                nconf = min(nv, npf)
                conf = nconf / (nconf + p["ratio_shrink"] + 1e-9)
                aux = aux + p["w_ratio"] * self.ratio_style * obs * conf

        if self.pos_pairs:
            early_z = []
            early_n = []
            late_z = []
            late_n = []
            for pos, vn, pn, vi, pi in self.pos_pairs:
                zs = []
                ns = []
                n1 = feats.n(vn)
                if n1 > 0.0:
                    z1 = (_logit_rate(feats.x(vn), n1) - self.f_mean[vi]) / self.f_std[vi]
                    zs.append(z1)
                    ns.append(n1)
                n2 = feats.n(pn)
                if n2 > 0.0:
                    z2 = (_logit_rate(feats.x(pn), n2) - self.f_mean[pi]) / self.f_std[pi]
                    zs.append(z2)
                    ns.append(n2)
                if zs:
                    zpos = sum(zs) / len(zs)
                    npos = sum(ns)
                    if pos in ("EP", "MP"):
                        early_z.append(zpos)
                        early_n.append(npos)
                    else:
                        late_z.append(zpos)
                        late_n.append(npos)
            if early_z and late_z:
                g = sum(late_z) / len(late_z) - sum(early_z) / len(early_z)
                g = (g - self.pos_offset) / (self.pos_scale + 1e-9)
                n_eff = min(sum(early_n), sum(late_n))
                conf = n_eff / (n_eff + p["posgrad_shrink"] + 1e-9)
                aux = aux + p["w_posgrad"] * self.pos_style * g * conf

        kappa = max(float(p["comp_kappa"]), 1e-3)
        mrate = np.clip(a / np.maximum(a + b, 1e-9), 1e-4, 1.0 - 1e-4)
        specs = (
            (("pfr",), ("vpip",), float(p["w_c_raise"])),
            (("limp",), ("vpip", "limp"), float(p["w_c_limp"])),
            (("vpip_CO", "vpip_BTN"),
             ("vpip_EP", "vpip_MP", "vpip_CO", "vpip_BTN"), float(p["w_c_pos"])),
        )
        for subs, bases, wc in specs:
            if wc <= 0.0:
                continue
            si = []
            for nm in subs:
                j = self.name_to_i.get(nm, -1)
                if j >= 0:
                    si.append(j)
            bi = []
            for nm in bases:
                j = self.name_to_i.get(nm, -1)
                if j >= 0:
                    bi.append(j)
            if not si or not bi:
                continue
            xs = 0.0
            nb = 0.0
            for j in si:
                xs += x[j]
            for j in bi:
                nb += x[j]
            if nb <= 0.0:
                continue
            if xs > nb:
                xs = nb
            num = mrate[:, si].sum(axis=1)
            den = mrate[:, bi].sum(axis=1)
            rr = np.clip(num / np.maximum(den, 1e-9), 0.02, 0.98)
            A = np.maximum(rr * kappa, 1e-6)
            B = np.maximum((1.0 - rr) * kappa, 1e-6)
            llc = (_lgamma(xs + A) + _lgamma(nb - xs + B)
                   - _lgamma(nb + A + B) - _log_beta(A, B))
            aux = aux + wc * llc

        w_ps = float(p["w_posshape"])
        if w_ps > 0.0:
            pos_items = []
            for pos in ("EP", "MP", "CO", "BTN", "SB", "BB"):
                vi = self.name_to_i.get("vpip_" + pos, -1)
                pi = self.name_to_i.get("pfr_" + pos, -1)
                if vi >= 0 or pi >= 0:
                    pos_items.append((vi, pi))
            if len(pos_items) >= 2:
                P = len(pos_items)
                style_raw = np.zeros((self.K, 2, P))
                pos_mask = np.zeros((2, P))
                obs = np.zeros((2, P))
                var = np.ones((2, P))
                cnt = np.zeros((2, P))
                for j, (vi, pi) in enumerate(pos_items):
                    if vi >= 0:
                        style_raw[:, 0, j] = self.means[:, vi]
                        pos_mask[0, j] = 1.0
                        if n[vi] > 0.0:
                            obs[0, j] = z[vi]
                            var[0, j] = s2[vi] + float(p["posshape_noise"])
                            cnt[0, j] = n[vi]
                    if pi >= 0:
                        style_raw[:, 1, j] = self.means[:, pi]
                        pos_mask[1, j] = 1.0
                        if n[pi] > 0.0:
                            obs[1, j] = z[pi]
                            var[1, j] = s2[pi] + float(p["posshape_noise"])
                            cnt[1, j] = n[pi]
                ll_ps = np.zeros(self.K)
                used_ps = False
                for t in range(2):
                    valid = (cnt[t] > 0.0) & (pos_mask[t] > 0.0)
                    if int(valid.sum()) >= 2:
                        oc_ps = obs[t, valid] - obs[t, valid].mean()
                        vv = var[t, valid]
                        for k in range(self.K):
                            sc = style_raw[k, t, valid]
                            sc = sc - sc.mean()
                            ll_ps[k] += -0.5 * np.sum((oc_ps - sc) ** 2 / vv)
                        used_ps = True
                if used_ps:
                    aux = aux + w_ps * ll_ps

        w_ds = float(p["w_depshape"])
        if w_ds > 0.0:
            dep_items = []
            for db in ("d0", "d1", "d2", "d3"):
                vi = self.name_to_i.get("vpip_" + db, -1)
                pi = self.name_to_i.get("pfr_" + db, -1)
                if vi >= 0 or pi >= 0:
                    dep_items.append((vi, pi))
            if len(dep_items) >= 2:
                P = len(dep_items)
                style_raw = np.zeros((self.K, 2, P))
                dep_mask = np.zeros((2, P))
                obs = np.zeros((2, P))
                var = np.ones((2, P))
                cnt = np.zeros((2, P))
                for j, (vi, pi) in enumerate(dep_items):
                    if vi >= 0:
                        style_raw[:, 0, j] = self.means[:, vi]
                        dep_mask[0, j] = 1.0
                        if n[vi] > 0.0:
                            obs[0, j] = z[vi]
                            var[0, j] = s2[vi] + float(p["depshape_noise"])
                            cnt[0, j] = n[vi]
                    if pi >= 0:
                        style_raw[:, 1, j] = self.means[:, pi]
                        dep_mask[1, j] = 1.0
                        if n[pi] > 0.0:
                            obs[1, j] = z[pi]
                            var[1, j] = s2[pi] + float(p["depshape_noise"])
                            cnt[1, j] = n[pi]
                ll_ds = np.zeros(self.K)
                used_ds = False
                for t in range(2):
                    valid = (cnt[t] > 0.0) & (dep_mask[t] > 0.0)
                    if int(valid.sum()) >= 2:
                        oc_ds = obs[t, valid] - obs[t, valid].mean()
                        vv = var[t, valid]
                        for k in range(self.K):
                            sc = style_raw[k, t, valid]
                            sc = sc - sc.mean()
                            ll_ds[k] += -0.5 * np.sum((oc_ds - sc) ** 2 / vv)
                        used_ds = True
                if used_ds:
                    aux = aux + w_ds * ll_ds

        w_bs = float(p["w_bshape"])
        if w_bs > 0.0:
            bs_names = ("betsize_b0", "betsize_b1", "betsize_b2", "betsize_b3",
                        "betsize_b4", "betsize_b5", "betsize_b6", "betsize_b7")
            bs_idx = []
            for nm in bs_names:
                j = self.name_to_i.get(nm, -1)
                if j >= 0:
                    bs_idx.append(j)
            if len(bs_idx) >= 2:
                xb = np.array([x[j] for j in bs_idx], dtype=float)
                Tb = xb.sum()
                if Tb > 0.0:
                    pr = np.clip(mrate[:, bs_idx], 1e-5, None)
                    pr = pr / np.maximum(pr.sum(axis=1, keepdims=True), 1e-9)
                    kap = max(float(p["bshape_kappa"]), 1e-3)
                    A = pr * kap
                    ll_bs = (_lgamma(kap) - _lgamma(Tb + kap)
                             + (_lgamma(xb[None, :] + A) - _lgamma(A)).sum(axis=1))
                    aux = aux + w_bs * ll_bs

        w_sdm = float(p["w_sdmean"])
        if w_sdm > 0.0:
            line_cnt = feats.shown.sum(axis=1)
            total_sd = line_cnt.sum()
            if total_sd > 0.0:
                bvec = np.arange(BUCKETS, dtype=float)
                obs_sum = (feats.shown * bvec[None, :]).sum()
                obs_mean = obs_sum / total_sd
                show = prof.showdown
                denom_sd = show.sum(axis=2)
                num_sd = (show * bvec[None, None, :]).sum(axis=2)
                exp_line = num_sd / np.maximum(denom_sd, 1e-12)
                exp_mean = (exp_line * line_cnt[None, :]).sum(axis=1) / total_sd
                noise_sd = max(float(p["sdmean_noise"]), 1e-3)
                ll_sdm = -0.5 * ((obs_mean - exp_mean) / noise_sd) ** 2
                aux = aux + w_sdm * ll_sdm

        prior = p["prior_power"] * self.log_shares
        base = prior + aux
        e_counts = p["w_bb"] * ll_bb
        e_gauss = p["w_gauss"] * ll_g + p["w_sd"] * sd
        tau = max(float(p["tau"]), 1e-3)
        a_mix = float(p["pool_mix"])
        a_mix = min(max(a_mix, 0.0), 1.0)
        if a_mix <= 0.0:
            return _softmax((base + e_counts + e_gauss) / tau)
        p_pool = _softmax((base + e_counts + e_gauss) / tau)
        p_cnt = _softmax((base + e_counts) / tau)
        p_gau = _softmax((base + e_gauss) / tau)
        mix = (1.0 - a_mix) * p_pool + 0.5 * a_mix * (p_cnt + p_gau)
        mix = np.maximum(mix, 1e-12)
        return mix / mix.sum()
        # END-EVOLVE-BLOCK


_UNUSED = (math, BUCKETS, LINES, HandEnd, HandStart, PlayerAction, Showdown, param)
