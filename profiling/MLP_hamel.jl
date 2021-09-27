cd(@__DIR__)
using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot
using UnPack
plotting = true

d = 1
x = fill(0.,d)  # initial point

tspan = (0e0,1e-2)
dt = 1e-2 # time step
μ(X,p,t) = 0e0 # advection coefficients
σ(X,p,t) = 1e-1 # diffusion coefficients
d = 5
ss0 = 1e-2#std g0
U = 5e-1
u_domain = repeat([-U,U]', d, 1)

σ_sampling = 1.
alg = MLP(M=4, K=10, L = 4, mc_sample = NormalSampling(σ_sampling) )


##########################
###### PDE Problem #######
##########################
g(x) = (2*π)^(-d/2) * ss0^(- Float64(d) * 5e-1) * exp.(-5e-1 *sum(x .^2e0 / ss0)) # initial condition
m(x) = - 5e-1 * sum(x.^2)
f(y, z, v_y, v_z, ∇v_y, ∇v_z, t) = max(0.0, v_y) * (m(y) - max(0.0, v_z) * m(z) * (2.0 * π)^(d/2) * σ_sampling^d * exp(0.5 * sum(z.^2) / σ_sampling^2)) # nonlocal nonlinear part of the

# defining the problem
prob = PIDEProblem(g, f, μ, σ, tspan, x = x)

# solving
@time xs,ts,sol = solve(prob, alg, verbose = false, multithreading=true)