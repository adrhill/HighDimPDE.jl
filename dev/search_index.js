var documenterSearchIndex = {"docs":
[{"location":"getting_started.html#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"The general workflow for using HighDimPDE.jl is as follows:","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Define a Partial Integro Differential Equation problem","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Modules = [HighDimPDE]\nPages   = [\"HighDimPDE.jl\"]","category":"page"},{"location":"getting_started.html#HighDimPDE.PIDEProblem","page":"Getting started","title":"HighDimPDE.PIDEProblem","text":"PIDEProblem(g, f, μ, σ, x, tspan, p = nothing, x=nothing, u_domain=nothing, neumann_bc=nothing)\n\nDefines a Partial Integro Differential Problem, of the form du/dt = 1/2 Tr(\\sigma \\sigma^T) Δu(t,x) + μ ∇u(t,x) + \\int f(u,x) dx; where f is a nonlinear Lipschitz function\n\nArguments\n\ng : The initial condition g(x, p, t).\nf : The function f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(y, t), p, t)\nμ : The drift function of X from Ito's Lemma μ(x, p, t)\nσ : The noise function of X from Ito's Lemma σ(x, p, t)\ntspan: The timespan of the problem.\np: the parameter \nx: the point of the solution required\nu_domain : if provided, approximating the solution on the hypercube u_domain[1] × u_domain[2]. \nneumann_bc: if provided, neumann boundary conditions on the hypercube neumann_bc[1] × neumann_bc[2]. \n\n\n\n\n\n","category":"type"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Select a solver algorithm\nSolve the problem","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"Let's solve the Fisher KPP PDE in dimension 10.","category":"page"},{"location":"getting_started.html#MLP-(@ref-mlp)","page":"Getting started","title":"MLP (@ref mlp)","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"partial_t u = u (1 - u) + frac12sigma^2Delta_xu tag1","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"using HighDimPDE\n\n## Definition of the problem\nd = 10 # dimension of the problem\ntspan = (0.0,0.5) # time horizon\nx0 = fill(0.,d)  # initial point\ng(x) = exp(- sum(x.^2) ) # initial condition\nμ(x, p, t) = 0.0 # advection coefficients\nσ(x, p, t) = 0.1 # diffusion coefficients\nf(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_x)) # nonlinear part of the PDE\nprob = PIDEProblem(g, f, μ, σ, x0, tspan) # defining the problem\n\n## Definition of the algorithm\nalg = MLP() # defining the algorithm. We use the Multi Level Picard algorithm\n\n## Solving with multiple threads \nsol = solve(prob, alg, multithreading=true)","category":"page"},{"location":"getting_started.html#DeepSplitting(@ref-deepsplitting)","page":"Getting started","title":"DeepSplitting(@ref deepsplitting)","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"using HighDimPDE\n\n## Definition of the problem\nd = 10 # dimension of the problem\ntspan = (0.0, 0.5) # time horizon\nx0 = fill(0.,d)  # initial point\ng(x) = exp.(- sum(x.^2, dims=1) ) # initial condition\nμ(x, p, t) = 0.0 # advection coefficients\nσ(x, p, t) = 0.1 # diffusion coefficients\nu_domain = [-1/2, 1/2]\nf(x, y, v_x, v_y, ∇v_x, ∇v_y, t) = max.(0f0, v_x) .* (1f0 .-  max.(0f0, v_x)) \nprob = PIDEProblem(g, f, μ, \n                    σ, x0, tspan, \n                    u_domain = u_domain)\n\n## Definition of the neural network to use\nusing Flux # needed to define the neural network\n\nhls = d + 50 #hidden layer size\n\nnn = Flux.Chain(Dense(d, hls, tanh),\n        Dense(hls, hls, tanh),\n        Dense(hls, 1)) # neural network used by the scheme\n\nopt = Flux.Optimiser(ExpDecay(0.1,\n                0.1,\n                200,\n                1e-4),\n                ADAM() )#optimiser\n\n## Definition of the algorithm\nalg = DeepSplitting(nn, opt = opt)\n\nsol = solve(prob, \n            alg, \n            dt=0.1, \n            verbose = true, \n            abstol = 2e-3,\n            maxiters = 1000,\n            batch_size = 1000)","category":"page"},{"location":"getting_started.html#Solving-on-the-GPU","page":"Getting started","title":"Solving on the GPU","text":"","category":"section"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"DeepSplitting can run on the GPU for (much) improved performance. To do so, just set use_cuda = true.","category":"page"},{"location":"getting_started.html","page":"Getting started","title":"Getting started","text":"sol = solve(prob, \n            alg, \n            dt=0.1, \n            verbose = true, \n            abstol = 2e-3,\n            maxiters = 1000,\n            batch_size = 1000,\n            use_cuda=true)","category":"page"},{"location":"Feynman_Kac.html#feynmankac","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"","category":"section"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"The Feynman Kac formula is generally stated for terminal condition problems (see e.g. Wikipedia), where","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"partial_t u(tx) + mu(x) nabla_x u(tx) + frac12 sigma^2(x) Delta_x u(tx) + f(x u(tx))  = 0 tag1","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"with terminal condition u(T x) = g(x), and u colon R^d to R. ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"In this case the FK formula states that for all t in (0T) it holds that","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"u(t x) = int_t^T mathbbE left f(X^x_s-t u(s X^x_s-t))ds right + mathbbE left u(0 X^x_T-t) right tag2","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"where ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"X_t^x = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + x","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"and B_t is a Brownian motion.","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"(Image: Brownian motion - Wikipedia)","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Intuitively, this formula is motivated by the fact that the density of Brownian particles (motion) satisfes the diffusion equation.","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"The equivalence between the average trajectory of particles and PDEs given by the Feynman-Kac formula allows to overcome the curse of dimensionality that standard numerical methods suffer from, because the expectations can be approximated Monte Carlo integrations, which approximation error decreases as 1sqrtN and is therefore not dependent on the dimensions. On the other hand, the computational complexity of traditional deterministic techniques grows exponentially in the number of dimensions. ","category":"page"},{"location":"Feynman_Kac.html#Forward-non-linear-Feynman-Kac","page":"Feynman Kac formula","title":"Forward non-linear Feynman-Kac","text":"","category":"section"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"How to transform previous equation to an initial value problem?","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Define v(tau x) = u(T-tau x). Observe that v(0x) = u(Tx). Further observe that by the chain rule","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"beginaligned\npartial_tau v(tau x) = partial_tau u(T-taux)\n                        = (partial_tau (T-tau)) partial_t u(T-taux)\n                        = -partial_t u(T-tau x)\nendaligned","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"From Eq. (1) we get that ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"- partial_t u(T - taux) = mu(x) nabla_x u(T - taux) + frac12 sigma^2(x) Delta_x u(T - taux) + f(x u(T - taux))","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"Replacing  u(T-tau x) by v(tau x) we get that v satisfies","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"partial_tau v(tau x) = mu(x) nabla_x v(taux) + frac12 sigma^2(x) Delta_x v(taux) + f(x v(taux)) ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"and from Eq. (2) we obtain","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"v(tau x) = int_T-tau^T mathbbE left f(X^x_s- T + tau v(s X^x_s-T + tau))ds right + mathbbE left v(0 X^x_tau) right","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"By using the substitution rule with tau to tau -T (shifting by T) and tau to - tau (inversing), and finally inversing the integral bound we get that ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"beginaligned\nv(tau x) = int_-tau^0 mathbbE left f(X^x_s + tau v(s + T X^x_s + tau))ds right + mathbbE left v(0 X^x_tau) right\n            = - int_tau^0 mathbbE left f(X^x_tau - s v(T-s X^x_tau - s))ds right + mathbbE left v(0 X^x_tau) right\n            = int_0^tau mathbbE left f(X^x_tau - s v(T-s X^x_tau - s))ds right + mathbbE left v(0 X^x_tau) right\nendaligned","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"This leads to the ","category":"page"},{"location":"Feynman_Kac.html","page":"Feynman Kac formula","title":"Feynman Kac formula","text":"info: Non-linear Feynman Kac for initial value problems\nConsider the PDEpartial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx))with initial conditions u(0 x) = g(x), where u colon R^d to R.  Thenu(t x) = int_0^t mathbbE left f(X^x_t - s u(T-s X^x_t - s))ds right + mathbbE left u(0 X^x_t) right tag3with X_t^x = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + x","category":"page"},{"location":"DeepSplitting.html#deepsplitting","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Modules = [HighDimPDE]\nPages   = [\"DeepSplitting.jl\"]","category":"page"},{"location":"DeepSplitting.html#HighDimPDE.DeepSplitting","page":"The DeepSplitting algorithm","title":"HighDimPDE.DeepSplitting","text":"DeepSplitting(nn; opt)\n\nDeep splitting algorithm.\n\nArguments\n\nnn: a Flux.Chain, or more generally a functor\nopt: optimiser to be use. By default, Flux.ADAM(0.1).\n\n\n\n\n\n","category":"type"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The DeepSplitting algorithm reformulates the PDE as a stochastic learning problem.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The algorithm relies on two main ideas:","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"the approximation of the solution u by a parametric function bf u^theta. This function is generally chosen as a (Feedforward) Neural Network, as it is a universal approximator.\nthe training of bf u^theta by simulated stochastic trajectories of particles, through the link between linear PDEs and the expected trajectory of associated Stochastic Differential Equations (SDEs), explicitly stated by the Feynman Kac formula.","category":"page"},{"location":"DeepSplitting.html#The-general-idea","page":"The DeepSplitting algorithm","title":"The general idea 💡","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Consider the PDE","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"partial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx)) tag1","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"with initial conditions u(0 x) = g(x), where u colon R^d to R. ","category":"page"},{"location":"DeepSplitting.html#Local-Feynman-Kac-formula","page":"The DeepSplitting algorithm","title":"Local Feynman Kac formula","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"DeepSplitting solves the PDE iteratively over small time intervals by using an approximate Feynman-Kac representation locally.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"More specifically, considering a small time step dt = t_n+1 - t_n one has that","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"u(t_n+1 X_T - t_n+1) approx mathbbE left f(t X_T - t_n u(t_nX_T - t_n))(t_n+1 - t_n) + u(t_n X_T - t_n)  X_T - t_n+1right tag3","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"One can therefore use Monte Carlo integrations to approximate the expectations","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"u(t_n+1 X_T - t_n+1) approx frac1textbatch_sizesum_j=1^textbatch_size left u(t_n X_T - t_n^(j)) + (t_n+1 - t_n)sum_k=1^K big f(t_n X_T - t_n^(j) u(t_nX_T - t_n^(j))) big right","category":"page"},{"location":"DeepSplitting.html#Reformulation-as-a-learning-problem","page":"The DeepSplitting algorithm","title":"Reformulation as a learning problem","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"The DeepSplitting algorithm approximates u(t_n+1 x) by a parametric function bf u^theta_n(x). It is advised to let this function be a neural network bf u_theta equiv NN_theta as they are universal approximators.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"For each time step t_n, the DeepSplitting algorithm ","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Generates the particle trajectories X^x (j) satisfying Eq. (2) over the whole interval 0T.\nSeeks bf u_n+1^theta  by minimising the loss function","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"L(theta) = bf u^theta_n+1(X_T - t_n) - left f(t X_T - t_n-1 bf u_n-1(X_T - t_n-1))(t_n - t_n-1) + bf u_n-1(X_T - t_n-1) right ","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"This way the PDE approximation problem is decomposed into a sequence of separate learning problems. In HighDimPDE.jl the right parameter combination theta is found by iteratively minimizing L using stochastic gradient descent.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"tip: Tip\nTo solve with DeepSplitting, one needs to provide to solvedt\nbatch_size\nmaxiters: the number of iterations for minimising the loss function\nabstol: the absolute tolerance for the loss function\nuse_cuda: if you have a Nvidia GPU, recommended.","category":"page"},{"location":"DeepSplitting.html#Solving-point-wise-or-on-a-hypercube","page":"The DeepSplitting algorithm","title":"Solving point-wise or on a hypercube","text":"","category":"section"},{"location":"DeepSplitting.html#Pointwise","page":"The DeepSplitting algorithm","title":"Pointwise","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"DeepSplitting allows to obtain u(tx) on a single point  x in Omega with the keyword x.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"prob = PIDEProblem(g, f, μ, σ, tspan, x = x)","category":"page"},{"location":"DeepSplitting.html#Hypercube","page":"The DeepSplitting algorithm","title":"Hypercube","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Yet more generally, one wants to solve Eq. (1) on a d-dimensional cube ab^d. This is offered by HighDimPDE.jl with the keyworkd u_domain.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"prob = PIDEProblem(g, f, μ, σ, tspan, u_domain = u_domain)","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Internally, this is handled by assigning a random variable as the initial point of the particles, i.e.","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"X_t^xi = int_0^t mu(X_s^x)ds + int_0^tsigma(X_s^x)dB_s + xi","category":"page"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"where xi a random variable uniformly distributed over ab^d. This way, the neural network is trained on the whole interval ab^d instead of a single point.","category":"page"},{"location":"DeepSplitting.html#References","page":"The DeepSplitting algorithm","title":"References","text":"","category":"section"},{"location":"DeepSplitting.html","page":"The DeepSplitting algorithm","title":"The DeepSplitting algorithm","text":"Beck, C., Becker, S., Cheridito, P., Jentzen, A., Neufeld, A., Deep splitting method for parabolic PDEs. arXiv (2019)\nHan, J., Jentzen, A., E, W., Solving high-dimensional partial differential equations using deep learning. arXiv (2018)","category":"page"},{"location":"MLP.html#mlp","page":"The MLP algorithm","title":"The MLP algorithm","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Modules = [HighDimPDE]\nPages   = [\"MLP.jl\"]","category":"page"},{"location":"MLP.html#HighDimPDE.MLP","page":"The MLP algorithm","title":"HighDimPDE.MLP","text":"MLP(;M=4, L=4,)\n\nMulti level Picard algorithm.\n\nArguments\n\nL: number of Picard iterations (Level),\nM: number of Monte Carlo integrations (at each level l, M^(L-l)integrations),\n\n\n\n\n\n","category":"type"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP, for Multi-Level Picard iterations, reformulates the PDE problem as a fixed point equation through the Feynman Kac formula. ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"It relies on Picard iterations to find the fixed point, \nreducing the complexity of the numerical approximation of the time integral through a multilevel Monte Carlo approach.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm overcomes the curse of dimensionality, with a computational complexity that grows polynomially in the number of dimension (see M. Hutzenthaler et al. 2020).","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"warning: `MLP` can only solve for one point at a time\nMLP works only with PIDEProblem defined with x = x option. If you want to solve over an entire domain, you definitely want to check the DeepSplitting algorithm.","category":"page"},{"location":"MLP.html#The-general-idea","page":"The MLP algorithm","title":"The general idea 💡","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Consider the PDE","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"partial_t u(tx) = mu(t x) nabla_x u(tx) + frac12 sigma^2(t x) Delta_x u(tx) + f(x u(tx)) tag1","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"with initial conditions u(0 x) = g(x), where u colon R^d to R. ","category":"page"},{"location":"MLP.html#Picard-Iterations","page":"The MLP algorithm","title":"Picard Iterations","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm observes that the Feynman Kac formula can be viewed as a fixed point equation, i.e. u = phi(u). Introducing a sequence (u_k) defined as u_0 = g and ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"u_l+1 = phi(u_l)","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"the Banach fixed-point theorem ensures that the sequence converges to the true solution u. Such a technique is known as Picard iterations.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The time integral term is evaluated by a Monte-Carlo integration","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"u_L  = frac1Msum_i^M left f(X^x(i)_t - s_(l i) u_L-1(T-s_i X^x( i)_t - s_(l i))) + u(0 X^x(i)_t - s_(l i)) right","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"But the MLP uses an extra trick to lower the computational cost of the iteration. ","category":"page"},{"location":"MLP.html#Telescope-sum","page":"The MLP algorithm","title":"Telescope sum","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"The MLP algorithm uses a telescope sum ","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"beginaligned\nu_L = phi(u_L-1) = phi(u_L-1) - phi(u_L-2) + phi(u_L-2) - phi(u_L-3) + dots \n= sum_l=1^L-1 phi(u_l-1) - phi(u_l-2)\nendaligned","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"As l grows, the term phi(u_l-1) - phi(u_l-2) becomes smaller - and demands more calculations. The MLP algorithm uses this fact by evaluating the integral term at level l with M^L-l samples.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"tip: Tip\nL corresponds to the level of the approximation, i.e. u approx u_L\nM characterises the number of samples for the monte carlo approximation of the time integral","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Overall, MLP can be summarised by the following formula","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"beginaligned\nu_L = sum_l=1^L-1 frac1M^L-lsum_i^M^L-l left f(X^x(l i)_t - s_(l i) u(T-s_(l i) X^x(l i)_t - s_(l i))) + mathbf1_N(l) f(X^x(l i)_t - s_(l i) u(T-s_(l i) X^x(l i)_t - s_(l i)))right\n\nqquad + frac1M^Lsum_i^M^L u(0 X^x(l i)_t)\nendaligned","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Note that the superscripts (l i) indicate the independence of the random variables X across levels.","category":"page"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"tip: Tip\nIn practice, if you have a non-local model you need to provide the sampling method and the number K of MC integration through the keywords mc_sample and K. K characterises the number of samples for the Monte Carlo approximation of the last term.\nmc_sample characterises the distribution of the Z variables","category":"page"},{"location":"MLP.html#References","page":"The MLP algorithm","title":"References","text":"","category":"section"},{"location":"MLP.html","page":"The MLP algorithm","title":"The MLP algorithm","text":"Becker, S., Braunwarth, R., Hutzenthaler, M., Jentzen, A., von Wurstemberger, P., Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations. arXiv (2020)","category":"page"},{"location":"index.html#HighDimPDE.jl","page":"Home","title":"HighDimPDE.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"HighDimPDE.jl is a Julia package to solve Highly Dimensional non-linear, non-local PDEs of the form","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"beginaligned\n   (partial_t u)(tx) =  fbig(tx u(tx) ( nabla_x u )(tx ) ) big)  + biglangle mu(tx) ( nabla_x u )( tx ) bigrangle \n     quad  + tfrac12 textTrace big(sigma(tx)  sigma(tx) ^* ( textHess_x u)(t x ) big)\nendaligned","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"where u colon 0T times Omega to R, Omega subset R^d, d large, subject to initial and boundary conditions.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"🚧 Work in Progress 🚧 for now, HighDimPDE.jl can only solve for local PDEs.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"HighDimPDE.jl implements solver algorithms that break down the curse of dimensionality, including","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"the Deep Splitting scheme\nthe Multi-Level Picard iterations scheme.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"To make the most out of HighDimPDE.jl, we advise to first have a look at the ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"documentation on the Feynman Kac formula,","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"as all solver algorithms heavily rely on it.","category":"page"},{"location":"index.html#Algorithm-overview","page":"Home","title":"Algorithm overview","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Features DeepSplitting MLP\nTime discretization free ❌ ✅\nMesh-free ✅ ✅\nSingle point x in R^d approximation ✅ ✅\nd-dimensional cube ab^d approximation ✅ ❌\nGPU ✅ ❌\nGradient non-linearities ✔️ ❌","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"✔️ : will be supported in the future","category":"page"}]
}
