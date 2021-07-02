using HighDimPDE
using Random
using Test
using Flux
using Revise
using PyPlot

# using the DeepSplitting alg
batch_size = 1000
train_steps = 2000
K = 10

tspan = (0.0,5f-1)
dt = 1f-1 # time step
μ(X,p,t) = 0f0 # advection coefficients
σ(X,p,t) = 0f0 # diffusion coefficients


u_domain = [-5f0,5f0]
d = 4

hls = d + 50 #hidden layer size

nn_batch = Flux.Chain(Dense(d,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,hls,tanh),
        BatchNorm(hls, affine=true),
        Dense(hls,1)) # Neural network used by the scheme, with batch normalisation

opt = Flux.Optimiser(ExpDecay(0.1,
                0.1,
                500,
                1e-6),
                ADAM() )#optimiser
alg = DeepSplitting(nn_batch, K=K, opt = opt,mc_sample = UniformSampling(u_domain[1],u_domain[2]) )


X0 = fill(0f0,d)  # initial point
g(X) = exp.(-0.25f0 * sum(X.^2,dims=1))   # initial condition
a(u) = u - u^3
f(y,z,v_y,v_z,∇v_y,∇v_z, t) = a.(v_y) .- a.(v_z) #.* Float32(π^(d/2)) * σ_sampling^d .* exp.(sum(z.^2, dims = 1) / σ_sampling^2) # nonlocal nonlinear part of the
# f(y,z,v_y,v_z,∇v_y,∇v_z, t) = zeros(Float32,size(v_y))

# defining the problem
prob = PIDEProblem(g, f, μ, σ, X0, tspan, 
                    u_domain = u_domain
                    )
# solving
@time xgrid, sol = solve(prob, 
                alg, 
                dt=dt, 
                verbose = true, 
                abstol=5f-5,
                maxiters = train_steps,
                batch_size=batch_size,
                use_cuda = true
                )

if d == 1
        plt.figure()
        for i in 1:length(sol)
                plt.scatter(reduce(vcat,xgrid), reduce(vcat,sol[i].(xgrid)))
        end
elseif d == 2
        plt.figure()
        xy = reduce(hcat,xgrid)
        plt.scatter(xy[1,:], xy[2,:], c = reduce(vcat,sol[1].(xgrid)))
else
        plt.figure()
        ts = 0.: dt : tspan[2]
        ys = [sol[i](zeros(d,1))[] for i in 1:length(sol)]
        plt.plot(collect(ts),ys)
end
gcf()