# Method

## Falsification problem

Each test is one closed-loop simulation of the AEB controller against a pedestrian
ahead in the same lane. A scenario is a point
$x = (v_e, d_0, v_p, \mu, \tau, \sigma) \in X \subset \mathbb{R}^6$: ego speed
$v_e \in [8, 20]$ m/s, initial gap $d_0$, pedestrian speed $v_p \in [0, 2]$ m/s,
road friction $\mu \in [0.35, 1]$, sensor delay $\tau \in [0, 0.5]$ s, and the standard
deviation of Gaussian range noise $\sigma \in [0, 1]$ m. The gap coordinate in $[8, 80]$ m
is rescaled onto $[d_{\min}(v_e, v_p, \mu), 80]$, where $d_{\min}$ is the smallest gap
at which ideal full braking from $t = 0$ satisfies every requirement. Every searched
scenario is therefore avoidable in principle, so a violation is due to the controller
and its degraded inputs, not to the initial conditions. Simulations use
$\Delta t = 0.05$ s over a 10 s horizon.

We monitor three STL safety requirements over the trace:

| Requirement | STL | Robustness |
|---|---|---|
| Collision avoidance | $\mathbf{G}\,(d > 0)$ | $\rho_c = \min_t d(t)$ |
| Minimum separation | $\mathbf{G}\,(d \ge 2\,\text{m})$ | $\rho_s = \min_t d(t) - 2$ |
| Time to collision | $\mathbf{G}\,(\mathrm{TTC} \ge 1.0\,\text{s})$ | $\rho_T = \min_t \mathrm{TTC}(t) - 1.0$ |

The aggregate robustness is $\rho = \min(\rho_c/1\,\text{m}, \rho_s/1\,\text{m}, \rho_T/1\,\text{s})$.
Because $\rho$ saturates once contact occurs, all search methods minimize the
severity-aware score

$$
f(x) = \begin{cases} -10 - v_{\text{impact}}(x) & \text{if a collision occurs,}\\ \rho(x) & \text{otherwise,}\end{cases}
$$

where $v_{\text{impact}}$ is ego speed at first contact (m/s). Lower $f$ means a worse
outcome, and every collision ranks below every non-collision.

## Failure archive and failure categories

A failure is any violated requirement. Each failure maps to one cell of a discrete
archive. The cell is defined by the most severe outcome (collision, then separation,
then TTC), a severity bin, a friction region, and a sensor region:

- Severity bins: impact speed in 5 m/s steps (4 bins), separation shortfall in
  0.5 m steps (4), TTC shortfall in 0.1 s steps (3).
- Friction region: low friction if $\mu < 0.6$, normal otherwise.
- Sensor region: degraded if $\tau \ge 0.25$ s or $\sigma \ge 0.5$ m, clean otherwise.

This gives $11 \times 2 \times 2 = 44$ cells. The archive keeps the most severe
scenario seen in each cell and never stores safe outcomes. *Failure coverage*, the
number of distinct cells reached, is our measure of failure diversity. Cells
describe outcomes and operating conditions; they are not established root causes.

## Robustness-guided baseline: CMA-ES with stall restarts

The baseline minimizes $f$ with CMA-ES (pycma 4.6.0). It works in coordinates
normalized to $[0,1]^6$, with pycma's bounded sampling, initial step size
$\sigma_0 = 0.2$, and the default population $\lambda = 4 + \lfloor 3 \ln 6 \rfloor = 9$.
The budget counts simulator calls exactly. A final partial generation is simulated but
not used to update the strategy.

CMA-ES converges to one region, so we restart it when it gets stuck. After each
complete generation, we compare that generation's median fitness with the best
median seen so far in the current run. A generation counts as stalled if it does not
improve on that best by at least $10^{-3}$ times its magnitude. The optimizer
restarts within the remaining budget when any of these holds:

1. Four consecutive generations stall.
2. Every coordinate's step size has fallen below $10^{-3}$ (converged).
3. A built-in pycma termination criterion fires.

We use the median rather than the best sample because a converging search keeps
lowering its median even in generations where no single sample beats the incumbent.
On a convex quadratic, a best-sample rule restarted runs that were still converging.
The median rule stays quiet there but fires on plateaus such as the saturated
maximum-impact corner of $X$. Without archive guidance, each restart draws a uniform
random mean with $\sigma_0 = 0.2$.

## Diversity-aware falsification: archive-guided CMA-ES

The archive-guided method is the *same* optimizer with two additions. Both read the
failure archive that is filled as simulations complete.

**Novelty fitness.** CMA-ES minimizes

$$
\tilde f(x) = f(x) - B \cdot \mathbb{1}[\,x \text{ reaches an archive cell not seen before}\,], \qquad B = 30.
$$

CMA-ES selection uses only ranks within a generation. Every failure has
$f \in [-30, 0)$ (impact speed is at most 20 m/s), and every other sample has
$f \ge -30$. So with $B = 30$, any sample that opens a new failure cell
($\tilde f < -30$) is ranked above every sample that does not, and the distribution mean moves toward unexplored failure
behaviour. Among samples that open no new cell, the ranking is still by severity. The
bonus is paid once per cell, so the objective stops rewarding a cell once it has
been found. On the tuning seeds, bonuses of 5 and 15, which are smaller than the
score range, gave lower coverage.

**Archive-seeded restarts.** When a run is stuck, then with probability 0.7 the next
restart is centred on the stored worst scenario of the least-visited archive cell
(ties broken uniformly), with $\sigma = 0.15$. Visits count every simulation that
lands in a cell. Otherwise the restart uses a uniform random mean, which keeps some
global exploration. Until the archive holds a failure, every restart is uniform.

The stall rule, its thresholds, the step sizes, and the population size are identical
to the baseline. With $B = 0$ and seeding disabled the method *is* the baseline:
it produces the same sequence of scenarios for the same seed.

## Sensor-noise seed protocol

Each simulation draws its range noise from a seeded generator. We give every
simulation its own seed, derived deterministically from the run seed and the
evaluation index (NumPy `SeedSequence([run_seed, i])`). All methods use this
protocol. Each saved scenario records its seed, so every reported failure replays
exactly.

The alternative, used before, reuses the run seed for every simulation in a run. Each
run then sees a single standard-normal noise sequence, scaled by $\sigma$. That lets a
search tune failures to one noise realization, and it means seeds vary the noise only
between runs, never within one. We compared the two protocols directly; the
ablation results report the comparison. To check that failures are not artefacts of
one noise draw, we re-simulate each archived worst failure with five unrelated noise
seeds and report the fraction that still fails.

## Ablation design

To isolate the archive's contribution, we run the same CMA-ES implementation in five
configurations. All share the same budget, seeds, noise protocol, and stall rule
unless a row says otherwise:

| Arm | Stall restarts | Novelty fitness | Archive restarts |
|---|:-:|:-:|:-:|
| CMA-ES, no stall restarts | – | – | – |
| CMA-ES (no archive), the baseline | ✓ | – | – |
| + novelty fitness | ✓ | ✓ | – |
| + archive restarts | ✓ | – | ✓ |
| + both (archive-guided CMA-ES) | ✓ | ✓ | ✓ |

Uniform random search and the archive mutation search (mutating the worst failure of
a uniformly chosen archive cell, with 20% uniform samples) are run alongside as
references.

Each arm runs on 30 seeds (6000–6029) at budgets of 100 and 300 simulations.
Settings were chosen on separate tuning seeds (5000–5029) and frozen before these runs.
Seeds 2000–2029 remain unused for the project's final comparison. Per run we record:

- failure coverage;
- worst impact speed (km/h, 0 if no collision);
- the number of collisions;
- the evaluation index of the first failure;
- restarts and archive-seeded restarts;
- the reseeding survival rate.

Each arm is compared with the no-archive baseline on the same seeds. For each pair we
report the mean paired difference, a 95% bootstrap confidence interval (20,000
resamples), and a two-sided paired sign-flip permutation test, with Holm correction
across arms within each metric.

The whole ablation, including the noise-seed comparison, runs with one command:

```bash
python -m aeb_falsification.ablation --noise per_evaluation fixed --output-dir experiment_results/ablation
```

## Limitations

The simulator is one-dimensional, the pedestrian moves longitudinally only, and
friction acts as a braking multiplier, not as a tyre model. Requirements are
monitored at sample times; collision contact is also detected within each step.
CMA-ES trajectories are bit-reproducible on one platform. Across platforms they drift
after the first covariance update because of floating-point differences in linear
algebra. Random search, the mutation search, and the replay of saved failures
reproduce exactly.
