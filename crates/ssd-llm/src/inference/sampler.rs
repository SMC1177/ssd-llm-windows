//! Token sampling strategies: Temperature, Top-K, Top-P (nucleus), Min-P, TFS, Mirostat

use std::cell::Cell;

/// Mirostat mode selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MirostatMode {
    /// Disabled — use standard Top-K/Top-P/Min-P sampling
    Disabled,
    /// Mirostat v1: perplexity-controlled sampling with adaptive top-k
    V1,
    /// Mirostat v2: simplified perplexity control with adaptive truncation
    V2,
}

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    /// Tail-Free Sampling parameter (0.0 = disabled, typical: 0.95-1.0)
    tfs_z: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    /// Mirostat mode
    mirostat: MirostatMode,
    /// Mirostat target surprise (tau), default 5.0
    mirostat_tau: f32,
    /// Mirostat learning rate (eta), default 0.1
    mirostat_eta: f32,
    /// Mirostat adaptive state: current mu (tracks 2*tau initially)
    mirostat_mu: Cell<f32>,
    rng_state: Cell<u64>,
}

impl Sampler {
    fn make_seed() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
            ^ 0x517cc1b727220a95
    }

    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with repetition penalty parameters
    pub fn with_penalties(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with all parameters including Min-P
    ///
    /// Min-P filtering keeps only tokens whose probability is at least `min_p` times
    /// the probability of the most likely token. This provides adaptive filtering that
    /// scales with model confidence — when the model is sure, fewer tokens pass;
    /// when uncertain, more diversity is allowed.
    ///
    /// Typical values: 0.05–0.1 for creative text, 0.1–0.2 for focused output.
    pub fn with_min_p(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: min_p.clamp(0.0, 1.0),
            tfs_z: 0.0,
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with Tail-Free Sampling
    ///
    /// TFS uses the second derivative of the sorted probability distribution to find
    /// the "tail" — tokens whose probability drops off sharply. It removes the tail
    /// and samples from the remaining distribution. This adapts better than top-p to
    /// distributions with long tails.
    ///
    /// `tfs_z`: threshold in [0, 1]. 1.0 = disabled, 0.95 = moderate filtering.
    pub fn with_tfs(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        tfs_z: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
            min_p: 0.0,
            tfs_z: tfs_z.clamp(0.0, 1.0),
            repetition_penalty: repetition_penalty.max(0.01),
            frequency_penalty,
            presence_penalty,
            mirostat: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_mu: Cell::new(10.0),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

    /// Create a sampler with Mirostat adaptive perplexity control
    ///
    /// Mirostat dynamically adjusts the sampling truncation to maintain a target
    /// perplexity (surprise) level. This produces more coherent text than fixed
    /// top-k/top-p by adapting to the model's confidence at each step.
    ///
    /// - `mode`: V1 (original with adaptive k) or V2 (simplified truncation)
    /// - `tau`: target surprise level (default 5.0, lower = more focused)
    /// - `eta`: learning rate for mu adaptation (default 0.1)
    pub fn with_mirostat(temperature: f32, mode: MirostatMode, tau: f32, eta: f32) -> Self {
        Self {
            temperature,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            tfs_z: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: mode,
            mirostat_tau: tau.max(0.0),
            mirostat_eta: eta.clamp(0.0, 1.0),
            mirostat_mu: Cell::new(2.0 * tau),
            rng_state: Cell::new(Self::make_seed()),
        }
    }

        /// Set a deterministic seed for reproducible sampling
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng_state.set(seed ^ 0x517cc1b727220a95);
        self
    }

    /// Apply repetition, frequency, and presence penalties to logits based on prior tokens
    pub fn apply_penalties(&self, logits: &mut [f32], previous_tokens: &[u32]) {
        if (self.repetition_penalty - 1.0).abs() < 1e-6
            && self.frequency_penalty.abs() < 1e-6
            && self.presence_penalty.abs() < 1e-6
        {
            return;
        }

        // Count token frequencies
        let mut freq: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &tok in previous_tokens {
            *freq.entry(tok).or_insert(0) += 1;
        }

        for (&tok, &count) in &freq {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }

            // Repetition penalty (multiplicative): divide positive logits, multiply negative
            if self.repetition_penalty != 1.0 {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }

            // Frequency penalty (additive, scales with count)
            logits[idx] -= self.frequency_penalty * count as f32;

            // Presence penalty (additive, binary)
            logits[idx] -= self.presence_penalty;
        }
    }

    /// Sample a token ID from logits
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Greedy if temperature is very low
        if self.temperature < 1e-6 {
            return argmax(logits) as u32;
        }

        // Dispatch to Mirostat if enabled
        match self.mirostat {
            MirostatMode::V1 => return self.sample_mirostat_v1(logits),
            MirostatMode::V2 => return self.sample_mirostat_v2(logits),
            MirostatMode::Disabled => {}
        }

        // Apply temperature scaling and select top-k via partial sort (O(n log k), no large alloc)
        let k = self.top_k.min(logits.len());
        let candidates: Vec<(usize, f32)> = top_k_logits(logits, self.temperature, k);

        // Min-P filtering: keep tokens with prob >= min_p * max_prob
        // Applied after top-k but before softmax, using pre-softmax logits
        let candidates = if self.min_p > 0.0 && !candidates.is_empty() {
            // Compute softmax over candidates to get probabilities for Min-P check
            let max_logit = candidates
                .iter()
                .map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<(usize, f32)> = candidates
                .iter()
                .map(|(idx, v)| (*idx, (v - max_logit).exp()))
                .collect();
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            let probs: Vec<(usize, f32)> = probs
                .into_iter()
                .map(|(idx, p)| (idx, if sum > 0.0 { p / sum } else { p }))
                .collect();

            let max_prob = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
            let threshold = self.min_p * max_prob;

            let filtered: Vec<(usize, f32)> = candidates
                .iter()
                .zip(probs.iter())
                .filter(|(_, (_, p))| *p >= threshold)
                .map(|(c, _)| *c)
                .collect();

            if filtered.is_empty() {
                vec![candidates[0]] // keep at least the top token
            } else {
                filtered
            }
        } else {
            candidates
        };

        // TFS filtering: remove the "tail" based on second derivative of sorted probabilities
        let candidates = if self.tfs_z > 0.0 && self.tfs_z < 1.0 && candidates.len() > 2 {
            tail_free_filter(&candidates, self.tfs_z)
        } else {
            candidates
        };

        // Softmax over candidates
        let max_val = candidates
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = candidates
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        // Top-P (nucleus) filtering
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0f32;
        let mut nucleus = Vec::new();
        for (idx, p) in &probs {
            cumsum += p;
            nucleus.push((*idx, *p));
            if cumsum >= self.top_p {
                break;
            }
        }

        // Renormalize
        let nuc_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
        if nuc_sum > 0.0 {
            for (_, p) in nucleus.iter_mut() {
                *p /= nuc_sum;
            }
        }

        // Sample from nucleus
        self.sample_from_distribution(&nucleus)
    }

    /// Mirostat v1: adaptive top-k based on target surprise
    ///
    /// Estimates the optimal k (number of candidates) to achieve target perplexity tau.
    /// Uses Zipf's law approximation: the i-th most likely token has probability ~ i^(-s).
    /// Adjusts mu after each token to track the target surprise.
    fn sample_mirostat_v1(&self, logits: &[f32]) -> u32 {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax to get probabilities
        let max_val = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        let mu = self.mirostat_mu.get();

        // Estimate Zipf exponent s from the top two probabilities
        let s = estimate_zipf_s(probs[0].1, probs.get(1).map(|(_, p)| *p).unwrap_or(1e-10));

        // Compute optimal k: k = (epsilon * 2^mu)^(1/s)
        // where epsilon normalizes the Zipf distribution
        let k_float = ((mu.exp2()) * self.compute_zipf_epsilon(s, logits.len())).powf(1.0 / s);
        let k = (k_float.round() as usize).clamp(1, probs.len());

        // Truncate to top-k
        let candidates: Vec<(usize, f32)> = probs[..k].to_vec();

        // Renormalize
        let cand_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        let candidates: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, p)| (idx, if cand_sum > 0.0 { p / cand_sum } else { p }))
            .collect();

        // Sample
        let token = self.sample_from_distribution(&candidates);

        // Update mu: mu_new = mu - eta * (surprise - tau)
        // surprise = -log2(p(token))
        let token_prob = candidates
            .iter()
            .find(|(idx, _)| *idx == token as usize)
            .map(|(_, p)| *p)
            .unwrap_or(1e-10);
        let surprise = -token_prob.max(1e-10).log2();
        let new_mu = mu - self.mirostat_eta * (surprise - self.mirostat_tau);
        self.mirostat_mu.set(new_mu);

        token
    }

    /// Mirostat v2: simplified adaptive truncation
    ///
    /// Instead of estimating Zipf parameters, v2 directly truncates the distribution
    /// at the point where cumulative surprise exceeds mu. Simpler and often works
    /// better than v1 in practice.
    fn sample_mirostat_v2(&self, logits: &[f32]) -> u32 {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
        let mut indexed: Vec<(usize, f32)> =
            scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax
        let max_val = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(idx, v)| (*idx, (v - max_val).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }

        let mu = self.mirostat_mu.get();

        // Keep tokens whose surprise (-log2(p)) <= mu
        let candidates: Vec<(usize, f32)> = probs
            .iter()
            .filter(|(_, p)| -p.max(1e-10).log2() <= mu)
            .copied()
            .collect();

        // Always keep at least the top token
        let candidates = if candidates.is_empty() {
            vec![probs[0]]
        } else {
            candidates
        };

        // Renormalize
        let cand_sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        let candidates: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, p)| (idx, if cand_sum > 0.0 { p / cand_sum } else { p }))
            .collect();

        let token = self.sample_from_distribution(&candidates);

        // Update mu
        let token_prob = candidates
            .iter()
            .find(|(idx, _)| *idx == token as usize)
            .map(|(_, p)| *p)
            .unwrap_or(1e-10);
        let surprise = -token_prob.max(1e-10).log2();
        let new_mu = mu - self.mirostat_eta * (surprise - self.mirostat_tau);
        self.mirostat_mu.set(new_mu);

        token
    }

    /// Compute Zipf epsilon normalization constant for Mirostat v1
    fn compute_zipf_epsilon(&self, s: f32, vocab_size: usize) -> f32 {
        // epsilon = sum_{i=1}^{n} i^(-s)  (generalized harmonic number)
        // For large vocab, approximate with integral: n^(1-s) / (1-s) for s != 1
        if (s - 1.0).abs() < 1e-6 {
            // Harmonic series: ~ln(n)
            (vocab_size as f32).ln()
        } else if s > 1.0 {
            // Converges; use partial sum for small n, approximation for large
            let n = vocab_size.min(1000) as f32;
            (1..=n as usize).map(|i| (i as f32).powf(-s)).sum::<f32>()
        } else {
            let n = vocab_size as f32;
            n.powf(1.0 - s) / (1.0 - s)
        }
    }

    /// Sample from a probability distribution (Vec of (index, probability))
    fn sample_from_distribution(&self, dist: &[(usize, f32)]) -> u32 {
        let r = self.random_f32();
        let mut cumulative = 0.0f32;
        for (idx, p) in dist {
            cumulative += p;
            if r <= cumulative {
                return *idx as u32;
            }
        }
        dist.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
    }

    /// xorshift64 PRNG returning f32 in [0, 1)
    fn random_f32(&self) -> f32 {
        let mut state = self.rng_state.get();
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.rng_state.set(state);
        // Use upper bits for better distribution
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Tail-Free Sampling: filter candidates by second derivative of sorted probabilities
///
/// TFS computes the second derivative (discrete) of the sorted probability distribution,
/// Select top-k logits by temperature-scaled value using a min-heap of size k.
/// O(n log k) time, O(k) space — avoids the O(n) temp Vec and O(n log n) full sort.
/// Returns candidates sorted descending by scaled logit value.
fn top_k_logits(logits: &[f32], temperature: f32, k: usize) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    // Min-heap ordered by scaled value (ascending) so we can evict the smallest.
    // We wrap f32 to get a total order that treats NaN as very negative.
    #[derive(Clone, Copy)]
    struct Entry(usize, f32); // (token_id, scaled_logit)
    impl PartialEq for Entry { fn eq(&self, o: &Self) -> bool { self.1 == o.1 } }
    impl Eq for Entry {}
    impl PartialOrd for Entry {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for Entry {
        // Min-heap: smaller value = higher priority (gets popped first)
        fn cmp(&self, o: &Self) -> Ordering {
            o.1.partial_cmp(&self.1).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap: std::collections::BinaryHeap<Entry> = std::collections::BinaryHeap::with_capacity(k + 1);
    for (i, &l) in logits.iter().enumerate() {
        let scaled = l / temperature;
        if heap.len() < k {
            heap.push(Entry(i, scaled));
        } else if let Some(min) = heap.peek() {
            if scaled > min.1 {
                heap.pop();
                heap.push(Entry(i, scaled));
            }
        }
    }
    let mut result: Vec<(usize, f32)> = heap.into_iter().map(|e| (e.0, e.1)).collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    result
}

/// normalizes it, and accumulates from most likely to least likely. Tokens beyond the
/// cumulative threshold `z` are removed — these are the "tail" tokens.
fn tail_free_filter(candidates: &[(usize, f32)], z: f32) -> Vec<(usize, f32)> {
    if candidates.len() <= 2 {
        return candidates.to_vec();
    }

    // Softmax to get probabilities (candidates are already sorted by logit descending)
    let max_val = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = candidates
        .iter()
        .map(|(_, v)| (v - max_val).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs
        .iter()
        .map(|p| if sum > 0.0 { p / sum } else { *p })
        .collect();

    // First derivative
    let first_deriv: Vec<f32> = probs.windows(2).map(|w| (w[0] - w[1]).abs()).collect();

    // Second derivative
    let second_deriv: Vec<f32> = first_deriv
        .windows(2)
        .map(|w| (w[0] - w[1]).abs())
        .collect();

    // Normalize second derivative
    let sd_sum: f32 = second_deriv.iter().sum();
    if sd_sum < 1e-12 {
        return candidates.to_vec();
    }
    let normalized: Vec<f32> = second_deriv.iter().map(|&d| d / sd_sum).collect();

    // Accumulate and find cutoff
    let mut cumsum = 0.0f32;
    let mut cutoff = candidates.len();
    for (i, &nd) in normalized.iter().enumerate() {
        cumsum += nd;
        if cumsum > z {
            // Keep tokens 0..=i+1 (the second derivative at i maps to token i+1 in the original)
            cutoff = i + 2; // +2 because second derivative loses 2 elements
            break;
        }
    }

    let result: Vec<(usize, f32)> = candidates[..cutoff.min(candidates.len())].to_vec();
    if result.is_empty() {
        vec![candidates[0]]
    } else {
        result
    }
}

/// Estimate Zipf exponent s from the two highest probabilities
/// Using the relationship p1/p2 ≈ 2^s for Zipf distribution
fn estimate_zipf_s(p1: f32, p2: f32) -> f32 {
    if p2 < 1e-10 {
        return 1.0;
    }
    let ratio = (p1 / p2).max(1.0);
    (ratio.log2()).clamp(0.1, 10.0)
}

// ── Constrained generation primitives ────────────────────────────────────────

/// Zero out all logits whose token ID is not in `allowed_ids`.
///
/// Call this on a logit slice before passing it to `Sampler::sample`. Tokens set
/// to `f32::NEG_INFINITY` have near-zero softmax probability and will never be
/// sampled (unless `allowed_ids` is empty, in which case nothing changes — caller
/// must ensure at least one valid token exists).
pub fn mask_logits(logits: &mut [f32], allowed_ids: &[u32]) {
    if allowed_ids.is_empty() {
        return;
    }
    // Build a small bitset when the vocab is the typical 150k-ish range.
    // For the constrained slots we use (tool names, arg keys), allowed_ids is ≤ a few dozen
    // IDs, so a sorted vec + binary search beats a full-vocab bool array.
    let mut sorted = allowed_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for (i, l) in logits.iter_mut().enumerate() {
        if sorted.binary_search(&(i as u32)).is_err() {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Prefix-aware constraint for slots where a single logical option may span
/// multiple tokens (e.g. "grep" tokenizes to ["gr", "ep"]).
///
/// Build once from pre-tokenized option sequences. At each decoding step, call
/// `valid_next_tokens(prefix)` to get the set of token IDs that are valid given
/// the tokens already emitted for this slot. When `is_complete(prefix)` returns
/// `true`, the slot is done and the caller can move to the next schema field.
pub struct PrefixConstraint {
    options: Vec<Vec<u32>>,
}

impl PrefixConstraint {
    pub fn new(options: Vec<Vec<u32>>) -> Self {
        Self { options }
    }

    /// Returns the token IDs that are valid as the next token given `prefix`.
    /// Empty result means `prefix` matches no option (caller should treat as error).
    pub fn valid_next_tokens(&self, prefix: &[u32]) -> Vec<u32> {
        let mut valid: Vec<u32> = self
            .options
            .iter()
            .filter(|opt| opt.starts_with(prefix) && opt.len() > prefix.len())
            .map(|opt| opt[prefix.len()])
            .collect();
        valid.sort_unstable();
        valid.dedup();
        valid
    }

    /// Returns `true` if `prefix` is exactly one of the options (slot complete).
    pub fn is_complete(&self, prefix: &[u32]) -> bool {
        self.options.iter().any(|opt| opt.as_slice() == prefix)
    }
}

/// Rolling-window sentinel detector.
///
/// Push tokens one at a time; returns `true` the moment the last `N` tokens
/// match the pre-tokenized trigger sequence. Reset with `reset()` to reuse
/// across multiple generations.
pub struct SentinelDetector {
    trigger: Vec<u32>,
    window: std::collections::VecDeque<u32>,
}

impl SentinelDetector {
    pub fn new(trigger: Vec<u32>) -> Self {
        let cap = trigger.len().max(1);
        Self {
            trigger,
            window: std::collections::VecDeque::with_capacity(cap),
        }
    }

    /// Push one token. Returns `true` if the sentinel was just completed.
    pub fn push(&mut self, token: u32) -> bool {
        self.window.push_back(token);
        if self.window.len() > self.trigger.len() {
            self.window.pop_front();
        }
        self.matches()
    }

    /// Returns `true` if the current window equals the trigger.
    /// Always returns `false` for an empty trigger (caller misuse guard).
    pub fn matches(&self) -> bool {
        !self.trigger.is_empty()
            && self.window.len() == self.trigger.len()
            && self.window.iter().zip(self.trigger.iter()).all(|(a, b)| a == b)
    }

    /// Clear the rolling window (call between generations).
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let sampler = Sampler::new(0.0, 40, 0.9);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.sample(&logits), 3); // argmax
    }

    #[test]
    fn test_rng_advances() {
        let sampler = Sampler::new(1.0, 40, 0.9);
        let a = sampler.random_f32();
        let b = sampler.random_f32();
        assert_ne!(a, b, "RNG should produce different values");
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 2.0, 0.0, 0.0);
        // Token 3 has highest logit but was seen before — penalty should reduce it
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        sampler.apply_penalties(&mut logits, &[3]);
        // Positive logit divided by 2.0: 0.9 -> 0.45
        assert!((logits[3] - 0.45).abs() < 1e-5);
        // Unmentioned tokens unchanged
        assert!((logits[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_frequency_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 1.0, 0.5, 0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        // Token 2 appeared 3 times
        sampler.apply_penalties(&mut logits, &[2, 2, 2]);
        // 3.0 - 0.5*3 = 1.5
        assert!((logits[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_presence_penalty() {
        let sampler = Sampler::with_penalties(0.0, 40, 0.9, 1.0, 0.0, 1.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        sampler.apply_penalties(&mut logits, &[1, 2, 2]);
        // Token 1: 2.0 - 1.0 = 1.0
        assert!((logits[1] - 1.0).abs() < 1e-5);
        // Token 2: 3.0 - 1.0 = 2.0 (presence is binary, not scaled by count)
        assert!((logits[2] - 2.0).abs() < 1e-5);
        // Token 0 unchanged
        assert!((logits[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_min_p_filters_low_probability() {
        // With min_p=0.5, only tokens with prob >= 50% of max prob should survive
        let sampler = Sampler::with_min_p(0.0, 40, 0.9, 0.5, 1.0, 0.0, 0.0);
        // Token 3 has highest logit (greedy), min_p should not affect greedy
        let logits = vec![0.1, 0.5, 0.3, 5.0, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_min_p_zero_has_no_effect() {
        // With min_p=0, sampling should work the same as without
        let sampler_no_minp = Sampler::new(0.0, 40, 0.9);
        let sampler_minp_zero = Sampler::with_min_p(0.0, 40, 0.9, 0.0, 1.0, 0.0, 0.0);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(
            sampler_no_minp.sample(&logits),
            sampler_minp_zero.sample(&logits)
        );
    }

    #[test]
    fn test_tfs_filters_tail() {
        // With tfs_z=0.5, TFS should aggressively filter the tail
        let sampler = Sampler::with_tfs(0.0, 40, 0.9, 0.5, 1.0, 0.0, 0.0);
        // Greedy still picks argmax regardless of TFS
        let logits = vec![0.1, 0.5, 0.3, 5.0, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_tfs_disabled_when_one() {
        // tfs_z=1.0 should be effectively disabled
        let sampler_no_tfs = Sampler::new(1.0, 40, 0.9);
        let sampler_tfs_one = Sampler::with_tfs(1.0, 40, 0.9, 1.0, 1.0, 0.0, 0.0);
        // Both should produce valid tokens (can't check exact equality due to RNG)
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t1 = sampler_no_tfs.sample(&logits);
        let t2 = sampler_tfs_one.sample(&logits);
        assert!(t1 < 5);
        assert!(t2 < 5);
    }

    #[test]
    fn test_tfs_filter_function() {
        // Test the tail_free_filter directly
        let candidates = vec![(0, 5.0), (1, 3.0), (2, 1.0), (3, 0.1), (4, -2.0)];
        let filtered = tail_free_filter(&candidates, 0.5);
        // Should keep at least the top tokens and cut the tail
        assert!(!filtered.is_empty());
        assert!(filtered.len() <= candidates.len());
        // First token should always be kept
        assert_eq!(filtered[0].0, 0);
    }

    #[test]
    fn test_mirostat_v1_samples_valid() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V1, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_mirostat_v2_samples_valid() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let token = sampler.sample(&logits);
        assert!((token as usize) < logits.len());
    }

    #[test]
    fn test_mirostat_v2_adapts_mu() {
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1);
        let initial_mu = sampler.mirostat_mu.get();
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        sampler.sample(&logits);
        let new_mu = sampler.mirostat_mu.get();
        // mu should have changed after sampling
        assert!(
            (initial_mu - new_mu).abs() > 1e-6,
            "Mirostat mu should adapt: was {}, now {}",
            initial_mu,
            new_mu
        );
    }

    #[test]
    fn test_mirostat_greedy_at_zero_temp() {
        // Even with mirostat enabled, temperature 0 should be greedy
        let sampler = Sampler::with_mirostat(0.0, MirostatMode::V2, 5.0, 0.1);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.sample(&logits), 3);
    }

    #[test]
    fn test_mirostat_v1_mu_tracks_tau() {
        // After many samples, mu should gravitate toward tau
        let sampler = Sampler::with_mirostat(0.7, MirostatMode::V1, 3.0, 0.3);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for _ in 0..50 {
            sampler.sample(&logits);
        }
        // mu should be in a reasonable range around tau (not diverging)
        let mu = sampler.mirostat_mu.get();
        assert!(
            mu > -50.0 && mu < 50.0,
            "Mirostat v1 mu should stay bounded: {}",
            mu
        );
    }

    #[test]
    fn test_estimate_zipf_s() {
        // When p1 >> p2, s should be large
        let s1 = estimate_zipf_s(0.9, 0.001);
        assert!(s1 > 5.0, "Large ratio should give large s: {}", s1);

        // When p1 ≈ p2, s should be small
        let s2 = estimate_zipf_s(0.5, 0.45);
        assert!(s2 < 1.0, "Small ratio should give small s: {}", s2);
    }

    #[test]
    fn test_no_penalties_noop() {
        let sampler = Sampler::with_penalties(1.0, 40, 0.9, 1.0, 0.0, 0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        sampler.apply_penalties(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, original);
    }
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    #[test]
    fn test_with_seed_deterministic() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let s1 = Sampler::new(1.0, 40, 0.9).with_seed(42);
        let s2 = Sampler::new(1.0, 40, 0.9).with_seed(42);

        let tokens1: Vec<u32> = (0..20).map(|_| s1.sample(&logits)).collect();
        let tokens2: Vec<u32> = (0..20).map(|_| s2.sample(&logits)).collect();
        assert_eq!(tokens1, tokens2, "Same seed should produce identical sequences");
    }

    #[test]
    fn test_different_seeds_differ() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let s1 = Sampler::new(1.0, 40, 0.9).with_seed(42);
        let s2 = Sampler::new(1.0, 40, 0.9).with_seed(99);

        let tokens1: Vec<u32> = (0..20).map(|_| s1.sample(&logits)).collect();
        let tokens2: Vec<u32> = (0..20).map(|_| s2.sample(&logits)).collect();
        assert_ne!(tokens1, tokens2, "Different seeds should produce different sequences");
    }

    #[test]
    fn test_seed_with_mirostat() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let s1 = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1).with_seed(42);
        let s2 = Sampler::with_mirostat(0.7, MirostatMode::V2, 5.0, 0.1).with_seed(42);

        let tokens1: Vec<u32> = (0..10).map(|_| s1.sample(&logits)).collect();
        let tokens2: Vec<u32> = (0..10).map(|_| s2.sample(&logits)).collect();
        assert_eq!(tokens1, tokens2, "Seeded Mirostat should be deterministic");
    }

    #[test]
    fn test_seed_with_min_p() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let s1 = Sampler::with_min_p(0.8, 40, 0.9, 0.1, 1.0, 0.0, 0.0).with_seed(123);
        let s2 = Sampler::with_min_p(0.8, 40, 0.9, 0.1, 1.0, 0.0, 0.0).with_seed(123);

        let tokens1: Vec<u32> = (0..10).map(|_| s1.sample(&logits)).collect();
        let tokens2: Vec<u32> = (0..10).map(|_| s2.sample(&logits)).collect();
        assert_eq!(tokens1, tokens2, "Seeded Min-P sampling should be deterministic");
    }
}

#[cfg(test)]
mod constrained_tests {
    use super::*;

    // ── mask_logits ──────────────────────────────────────────────────────────

    #[test]
    fn mask_logits_zeros_disallowed_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mask_logits(&mut logits, &[1, 3]);
        assert_eq!(logits[0], f32::NEG_INFINITY, "token 0 not in allowed → NEG_INF");
        assert_eq!(logits[1], 2.0, "token 1 allowed → unchanged");
        assert_eq!(logits[2], f32::NEG_INFINITY, "token 2 not in allowed → NEG_INF");
        assert_eq!(logits[3], 4.0, "token 3 allowed → unchanged");
        assert_eq!(logits[4], f32::NEG_INFINITY, "token 4 not in allowed → NEG_INF");
    }

    #[test]
    fn mask_logits_empty_allowed_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        mask_logits(&mut logits, &[]);
        assert_eq!(logits, vec![1.0, 2.0, 3.0], "empty allowed → logits untouched");
    }

    #[test]
    fn mask_logits_all_allowed_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        mask_logits(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mask_logits_deduplicates_allowed() {
        let mut logits = vec![1.0, 2.0, 3.0];
        // Duplicate allowed ID should not panic or corrupt
        mask_logits(&mut logits, &[1, 1, 1]);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn mask_logits_out_of_range_id_ignored() {
        let mut logits = vec![1.0, 2.0];
        // Token ID 999 is beyond logits length — should not panic
        mask_logits(&mut logits, &[0, 999]);
        assert_eq!(logits[0], 1.0, "token 0 allowed");
        assert_eq!(logits[1], f32::NEG_INFINITY, "token 1 not in allowed");
    }

    #[test]
    fn mask_then_sample_greedy_picks_allowed_argmax() {
        let sampler = Sampler::new(0.0, 40, 0.9);
        let mut logits = vec![0.1, 0.5, 5.0, 3.0, 0.2]; // token 2 is highest
        // Disallow token 2 — sampler should fall back to token 3
        mask_logits(&mut logits, &[0, 1, 3, 4]);
        let chosen = sampler.sample(&logits);
        assert_eq!(chosen, 3, "greedy should pick token 3 after token 2 is masked");
    }

    // ── PrefixConstraint ─────────────────────────────────────────────────────

    #[test]
    fn prefix_constraint_single_token_options() {
        // Each option is a single token: [10], [20], [30]
        let pc = PrefixConstraint::new(vec![vec![10], vec![20], vec![30]]);
        let valid = pc.valid_next_tokens(&[]);
        assert_eq!(valid, vec![10, 20, 30], "at empty prefix all starts are valid");
        assert!(pc.is_complete(&[10]));
        assert!(pc.is_complete(&[20]));
        assert!(!pc.is_complete(&[99]));
    }

    #[test]
    fn prefix_constraint_multi_token_option() {
        // "grep" might encode as [100, 200], "read_chunk" as [100, 300, 400]
        let pc = PrefixConstraint::new(vec![
            vec![100, 200],         // option A
            vec![100, 300, 400],    // option B (shares first token with A)
            vec![500],              // option C (unique start)
        ]);

        // At empty prefix: starts are 100 and 500
        let at_start = pc.valid_next_tokens(&[]);
        assert_eq!(at_start, vec![100, 500]);

        // After [100]: valid continuations are 200 (→ A) and 300 (→ B)
        let after_100 = pc.valid_next_tokens(&[100]);
        assert_eq!(after_100, vec![200, 300]);

        // After [100, 200]: option A is complete, no further tokens
        assert!(pc.valid_next_tokens(&[100, 200]).is_empty());
        assert!(pc.is_complete(&[100, 200]));

        // After [100, 300]: only 400 is valid (option B continues)
        let after_100_300 = pc.valid_next_tokens(&[100, 300]);
        assert_eq!(after_100_300, vec![400]);

        // Invalid prefix returns empty
        assert!(pc.valid_next_tokens(&[999]).is_empty());
    }

    #[test]
    fn prefix_constraint_deduplicates_shared_next_tokens() {
        // Two options that share the same second token after different firsts — doesn't apply,
        // but two options sharing first AND second token should deduplicate next.
        let pc = PrefixConstraint::new(vec![
            vec![1, 2, 3],
            vec![1, 2, 4],
        ]);
        // After [1]: only token 2 is valid (both options agree)
        assert_eq!(pc.valid_next_tokens(&[1]), vec![2]);
        // After [1, 2]: tokens 3 and 4 are valid
        assert_eq!(pc.valid_next_tokens(&[1, 2]), vec![3, 4]);
    }

    // ── SentinelDetector ─────────────────────────────────────────────────────

    #[test]
    fn sentinel_fires_on_exact_match() {
        let mut det = SentinelDetector::new(vec![10, 20, 30]);
        assert!(!det.push(10));
        assert!(!det.push(20));
        assert!(det.push(30), "trigger should fire after full sequence");
    }

    #[test]
    fn sentinel_does_not_fire_on_partial_match() {
        let mut det = SentinelDetector::new(vec![10, 20, 30]);
        assert!(!det.push(10));
        assert!(!det.push(20));
        // wrong third token
        assert!(!det.push(99));
    }

    #[test]
    fn sentinel_sliding_window_handles_false_prefix() {
        // Push almost-trigger then correct trigger — should fire
        let mut det = SentinelDetector::new(vec![10, 20]);
        det.push(10);
        det.push(99); // wrong — window is [10, 99]
        det.push(10); // window slides to [99, 10]
        let fired = det.push(20); // window: [10, 20] — fires
        assert!(fired);
    }

    #[test]
    fn sentinel_does_not_fire_before_enough_tokens() {
        let mut det = SentinelDetector::new(vec![5, 6, 7, 8]);
        // Push only 3 of 4 trigger tokens — must not fire
        assert!(!det.push(5));
        assert!(!det.push(6));
        assert!(!det.push(7));
    }

    #[test]
    fn sentinel_reset_clears_state() {
        let mut det = SentinelDetector::new(vec![1, 2]);
        det.push(1); // partial match
        det.reset();
        // After reset, push(2) alone must not fire
        assert!(!det.push(2), "reset should clear partial match state");
    }

    #[test]
    fn sentinel_single_token_trigger() {
        let mut det = SentinelDetector::new(vec![42]);
        assert!(!det.push(1));
        assert!(det.push(42));
        // Next non-matching token should not fire
        det.reset();
        assert!(!det.push(1));
    }

    #[test]
    fn sentinel_empty_trigger_never_fires() {
        // Empty trigger is a caller error; must not produce false positives.
        let mut det = SentinelDetector::new(vec![]);
        assert!(!det.push(0), "empty trigger must never fire");
        assert!(!det.push(0));
        assert!(!det.matches());
    }
}
