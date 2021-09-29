# The `MLP` algorithm

```@autodocs
Modules = [HighDimPDE]
Pages   = ["MLP.jl"]
```

The `MLP`, for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. 

- It relies on [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem) to find the fixed point, 

- reducing the complexity of the numerical approximation of the time integral through a [multilvel Monte Carlo](https://en.wikipedia.org/wiki/Multilevel_Monte_Carlo_method) approach.

The `MLP` algorithm overcomes the curse of dimensionality, with a computational complexity that grows polynomially in the number of dimension (see [M. Hutzenthaler et al. 2020](https://arxiv.org/abs/1807.01212v3)).

!!! warning "`MLP` can only solve for one point at a time"
    `MLP` works only with `PIDEProblem` defined with `x = x` option). If you want to solve over an entire domain, you definitely want to check the `DeepSplitting` algorithm.

## The general idea 💡
Consider the PDE
```math
\partial_t u(t,x) = \mu(t, x) \nabla_x u(t,x) + \frac{1}{2} \sigma^2(t, x) \Delta_x u(t,x) + f(x, u(t,x))
```
with initial conditions $u(0, x) = g(x)$, where $u \colon \R^d \to \R$. 

Recall that the nonlinear Feynman-Kac formula provides a solution in terms of the mean trajectory of the stochastic trajectory of particles  $X^x_t$ 
```math
u(t, x) = \int_0^t \mathbb{E} \left[ f(X^x_{t - s}, u(T-s, X^x_{t - s}))ds \right] + \mathbb{E} \left[ u(0, X^x_t) \right]
```
where 
```math
X_t^x = \int_0^t \mu(X_s^x)ds + \int_0^t\sigma(X_s^x)dB_s + x,
```

> The Feynman Kac formula is often expressed for terminal condition problems where $u(T,x) = g(x)$. See Ref. for the equivalence between initial condition problems $u(0,x) = g(x)$.

### Picard Iterations
The `MLP` algorithm observes that Eq. (1_ can be viewed as a fixed point equation, i.e. $u = \phi(u)$. Introducing a sequence $(u_k)$ defined as $u_0 = g$ and 
```
u_{l+1} = \phi(u_l),
```
the [Banach fixed-point theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) ensures that the sequence converges to the true solution $u$.

Such a technique is known as [Picard iterations](https://en.wikipedia.org/wiki/Picard–Lindelöf_theorem)


The integral term can be evaluated by a plain vanilla [Monte-Carlo integration]()

```math
u_L  = \frac{1}{M}\sum_i^M \mathbb{E} \left[ f(X^x_{t - s_i}, u_{L-1}(T-s_i, X^x_{t - s_i})) \right] + \mathbb{E} \left[ u(0, X^x_t) \right].
```

But the MLP uses an extra trick to lower the computational cost of the iteration. 


### Telescope sum
The `MLP` algorithm uses a telescope sum 

```math
\begin{aligned}
u_L = \phi(u_{L-1}) &= [\phi(u_{L-1}) - \phi(u_{L-2})] + [\phi(u_{L-2}) - \phi(u_{L-3})] + \dots \\
&= \sum_{l=1}^{L-1} [\phi(u_{l-1}) - \phi(u_{l-2})]
\end{aligned}
```

As $l$ grows, the term $[\phi(u_{l-1}) - \phi(u_{l-2})]$ becomes smaller - and demands more calculations. The `MLP` algorithm usses this fact by evaluating the integral term at level $l$ with $M^{L-l}$ samples.


!!! tip
    - `L` corresponds to the level of the approximation, i.e. $u \approx u_L$
    - `M` characterises the number of samples for the monte carlo approximation of the time integral

```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_i^{M^{L-l}} \left[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}})) + \1_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\right]
\\
&\qquad + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```

## Accounting for non-localness
Similar to the `DeepSplitting` algorithm, `MLP` offers to solve for non-local reaction diffusion equations of the type
```math
\partial_t u = \mu(t, x) \nabla_x u(t, x) + \frac{1}{2} \sigma^2(t, x) \Delta u(t, x) + \int_{\Omega}f(x, y, u(t,x), u(t,y))dy
```

The non-localness is again handled by a plain vanilla Monte Carlo integration.

```math
\begin{aligned}
u_L &= \sum_{l=1}^{L-1} \frac{1}{M^{L-l}}\sum_{i=1}^{M^{L-l}} \frac{1}{K}\sum_{j=1}^{K}  \bigg[ f(X^{x,(l, i)}_{t - s_{(l, i)}}, Z^{(l,j)}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}), u(T-s_{l,i}, Z^{(l,j)})) + \\
&\qquad 
\1_\N(l) f(X^{x,(l, i)}_{t - s_{(l, i)}}, u(T-s_{(l, i)}, X^{x,(l, i)}_{t - s_{(l, i)}}))\bigg] + \frac{1}{M^{L}}\sum_i^{M^{L}} u(0, X^{x,(l, i)}_t)\\
\end{aligned}
```

!!! tip
    - `K` characterises the number of samples for the Monte Carlo approximation of the last term.
    - `mc_sample` characterises the distribution of the `Z` variables

### References