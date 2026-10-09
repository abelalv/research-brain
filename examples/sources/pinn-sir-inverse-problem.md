# Physics-informed neural networks for estimating transmission rates in SIR models

_A short teaching text written for the research-brain demo. It is not a published paper._

## 1. Problem

The SIR model divides a closed population of size $N$ into susceptible $S(t)$, infected
$I(t)$ and recovered $R(t)$ individuals:

$$
\frac{dS}{dt} = -\beta \frac{S I}{N}, \qquad
\frac{dI}{dt} = \beta \frac{S I}{N} - \gamma I, \qquad
\frac{dR}{dt} = \gamma I ,
$$

where $\beta$ is the transmission rate and $\gamma$ the recovery rate. In practice neither
rate is known: they must be estimated from case counts. Estimating model coefficients from
observations of the solution is an inverse problem, and it is often ill-posed — different
parameter values can produce nearly the same observed curve.

## 2. Method

A physics-informed neural network (PINN) approximates the state $(S, I, R)$ with a neural
network $u_\theta(t)$ and treats $\beta$ (and possibly $\gamma$) as additional trainable
parameters. Training minimises

$$
\mathcal{L}(\theta, \beta) = \mathcal{L}_{\text{data}} + \lambda_r \, \mathcal{L}_{\text{res}} + \lambda_0 \, \mathcal{L}_{\text{IC}},
$$

where $\mathcal{L}_{\text{data}}$ is the misfit to observed incidence, $\mathcal{L}_{\text{res}}$
is the mean squared residual of the three ODEs evaluated at a finite set of collocation
points $\{t_k\}_{k=1}^{K}$, and $\mathcal{L}_{\text{IC}}$ penalises deviations from the
initial conditions. The weights $\lambda_r, \lambda_0$ are chosen by hand and the results
are sensitive to them.

## 3. Results

In a synthetic experiment with $N = 10^5$, $\beta = 0.3$ and $\gamma = 0.1$ per day, the
PINN recovered $\beta$ within 5% of its true value from 20 noisy observations of $I(t)$
(10% multiplicative noise). With a time-varying $\beta(t)$ represented by a second
network, the estimates became unstable unless a smoothness penalty was added.

## 4. Limitations

The physics loss is only enforced at the collocation points, so a small training loss does
not guarantee that the residual vanishes on the whole interval. When only reported cases
$\rho I(t)$ are observed with an unknown reporting fraction $\rho$, the reporting fraction
and the initial number of infected individuals cannot be separated from early-epidemic
data: they are not jointly identifiable.

## 5. Open question

Can the approach be extended to metapopulation models on a contact network, where each
node has its own transmission rate and nodes are coupled through the graph Laplacian?
Graph neural networks are a natural candidate for representing such spatially coupled
states.
