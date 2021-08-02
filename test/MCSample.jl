using HighDimPDE, Random, Test, CUDA

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU MCSampling" begin 
        # uniform sampling
        mc_sample = UniformSampling(-1f0, 1f0)
        X = CUDA.zeros(Float32,10,8000)
        XX = deepcopy(X)
        mc_sample(X)
        @test size(X) == size(XX)
        @test typeof(X) == typeof(XX)
        @test all( -1f0 .< X .< 1f0 )

        # uniform partial sampling
        mc_sample = UniformSampling(-1f0,1f0, [1])
        X = CUDA.ones(Float32, 10,8000)
        XX = deepcopy(X)
        mc_sample(X, XX)
        @test size(X) == size(XX)
        @test typeof(X) == typeof(XX)
        @test all(X[1,:] .== 1.) && all( (X[2:end,:] .!= 1))

        # normal sampling
        mc_sample = NormalSampling(1f0,false)
        X = CUDA.ones(Float32, 10,8000)
        XX = deepcopy(X)
        mc_sample(X)
        @test size(X) == size(XX)
        @test typeof(X) == typeof(XX)

        # normal partial sampling
        mc_sample = NormalSampling(1f0, false, [1])
        X = CUDA.ones(Float32, 10,8000)
        XX = deepcopy(X)
        mc_sample(X, XX)
        @test size(X) == size(XX)
        @test typeof(X) == typeof(XX)
        @test all(X[1,:] .== 1.) && all( (X[2:end,:] .!= 1))

        # normal sampling - true
        mc_sample = NormalSampling(1.,true)
        X = CUDA.ones(Float32, 10,8000)
        XX = deepcopy(X)
        mc_sample(X, XX)
        @test size(X) == size(XX)
        @test typeof(X) == typeof(XX)
    end
end

@testset "CPU MCSampling" begin 
    # uniform sampling
    mc_sample = UniformSampling(0.,1.)
    X = zeros(10,8000)
    XX = deepcopy(X)
    mc_sample(X)
    @test size(X) == size(XX)
    @test typeof(X) == typeof(XX)
    @test all( 0f0 .< X .< 1f0 )
    
    # uniform partial sampling
    mc_sample = UniformSampling(-1f0, 1f0, [1])
    X = ones(Float32, 10,8000)
    XX = deepcopy(X)
    mc_sample(X, XX)
    @test size(X) == size(XX)
    @test typeof(X) == typeof(XX)
    @test all(X[1,:] .== 1.) && all(X[2:end,:] .!= 1)

    # normal sampling
    mc_sample = NormalSampling(1., false)
    X = zeros(10,8000)
    XX = deepcopy(X)
    mc_sample(X)
    @test size(X) == size(XX)
    @test typeof(X) == typeof(XX)

    # normal partial sampling
    mc_sample = NormalSampling(1f0, false, [1])
    X = ones(Float32, 10,8000)
    XX = deepcopy(X)
    mc_sample(X, XX)
    @test size(X) == size(XX)
    @test typeof(X) == typeof(XX)
    @test all(X[1,:] .== 1.) && all((X[2:end,:] .!= 1))

    # normal sampling - true
    mc_sample = NormalSampling(1.,true)
    X = ones(10,8000)
    XX = deepcopy(X)
    mc_sample(X, XX)
    @test size(X) == size(XX)
    @test typeof(X) == typeof(XX)
end
