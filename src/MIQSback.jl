module MIQSback

using LinearAlgebra
using Zygote
using Optim
using GellMannMatrices
using RandomMatrices
using StaticArrays
using Tullio

struct System
    dim::Int64
    anc_dim::Int64
    rho_d
    su
    projectors
    total_dim
    function System(dim::Int64, anc_dim::Int64=2, rho_d=[1.0+0im 0; 0 0])

        # Construct lie algebra basis
        total_dim = (anc_dim*dim)^2 - 1
        su = gellmann(anc_dim * dim)

        # Construct projection operators for measurement
        anc_proj = zeros(anc_dim, anc_dim, anc_dim)
        for i in 1:anc_dim
            anc_proj[i,i,i] = 1
        end
        projectors = zeros(anc_dim*dim, anc_dim*dim, anc_dim)
        for i in 1:anc_dim
            projectors[:,:,i] = kron(anc_proj[:,:,i], I(dim))
        end
        
        new(dim, anc_dim, rho_d, su, projectors, total_dim)
    end
end

function U_to_param(s::System, U::Matrix{ComplexF64})
    H = 1im * log(U)
    x = real(tr.(Ref(H) .* s.su)) / 2
    return x
end

function param_to_U(s::System, x::AbstractArray, noise_level = 0)
    # H = sum((x .+ noise_level * error) .* s.su)

    H = sum(x  .* s.su)

    U = exp(-1im * H)
    return U
end

function rand_param(s::System)
    dim = s.dim
    anc_dim = s.anc_dim
    
    tree = U_to_param(s, rand(Haar(2), anc_dim * dim))
    return tree;
end

function total_tree_count(r, N)
    return Int((1 - r^N)/(1-r))
end

function compute_tree_depth(r, X)
    return Int(log((r-1) * X + 1) / log(r))
end

function compute_one_time(s::System, params, rho_r, noise_level)

    # U_r = param_to_U.(Ref(s), params)
    U_r = param_to_U.(Ref(s), eachcol(params), Ref(noise_level))
    U_r = reduce(hcat, U_r)
    U_r = reshape(U_r, (s.dim*s.anc_dim,s.dim*s.anc_dim,size(params,2)))

    # Unitaries for a given "time"

    # Action of unitary + measurement stored as [:,:,P,U]
    @tullio K[i,k,c,d] := s.projectors[i,j,c] *  U_r[j,k,d]

    # println("projector $(size(s.projectors))")
    # println("U_r $(size(U_r))")
    # println("K $(size(K))")
    # println("rho_r $(size(rho_r))")

    K_conj = conj(K)

    @tullio rho_n[i,j,c,d] := rho_r[i,k,d] * K_conj[j,k,c,d]
    @tullio rho_m[i,k,c,d] := K[i,j,c,d] * rho_n[j,k,c,d]

    # @assert K[:,:,1,1] * rho_r[:,:,1] * K[:,:,1,1]' ≈ rho_m[:,:,1,1]
    # @assert K[:,:,2,1] * rho_r[:,:,1] * K[:,:,2,1]' ≈ rho_m[:,:,2,1]
    @tullio prob[i,k,c,d] := rho_m[i,j,c,d] * s.projectors[j,k,c]
    @tullio prob_out[c,d] := prob[i,i,c,d]
    prob_out = real(prob_out)
    @tullio rho_out[i,j,c,d] := rho_m[i,j,c,d] / prob_out[c,d]


    # compute the partial trace
    tensor = reshape(rho_out, (s.dim,s.anc_dim,s.dim,s.anc_dim,s.anc_dim,size(params,2)))
    @tullio rho_system[j,k,c,d] := tensor[j,i,k,i,c,d]
    return prob_out, rho_system;

    # target_conj = conj(target)

    # return sum(F.(eachslice(reshape(rho_system, (2,2,6));dims=3)))

    # @tullio fidelity[c,d] := target_conj[i] * rho_system[i,j,c,d] * target[i]
    # fidelity[1,2] ≈ target' * rho_system[:,:,1,2] * target
    # fidelity[2,2] ≈ target' * rho_system[:,:,2,2] * target

    # take fidelity and compute the sum
    # @tullio fidelity := target_conj[j] * rho_system[i,j,c,d] * target[i]
    # return real(fidelity)
end

function forward_pass(s::System, target::Matrix{ComplexF64}, params::Matrix{Float64}, rho_s_i::Matrix{ComplexF64}, noise_level; individual_fidelity=false, individual_probability=false, individual_rho=false)
    iter = compute_tree_depth(s.anc_dim, size(params, 2))
    # println(iter)
    rho = kron(s.rho_d, rho_s_i)
    rho = reshape(rho, (s.dim*s.anc_dim, s.dim*s.anc_dim, 1))
    probs, rho_s = compute_one_time(s, params[:,1], rho, noise_level)
    # Tensor product
    @tullio rho[a,k,c,p,b,d] := rho_s[k,p,b,d] * s.rho_d[a,c]
    rho = reshape(rho, s.dim*s.anc_dim,s.dim*s.anc_dim,s.anc_dim)

    # println(size(rho))

    for i in 1:iter-1
        probs, rho_s = compute_one_time(s, params[:, total_tree_count(s.anc_dim,i)+1:total_tree_count(s.anc_dim,i+1)], rho, noise_level)
        # Tensor product
        @tullio rho[a,k,c,p,b,d] := rho_s[k,p,b,d] * s.rho_d[a,c]
        rho = reshape(rho, s.dim*s.anc_dim,s.dim*s.anc_dim,s.anc_dim^(i+1))
    end

    if individual_rho
        return rho_s;
    end

    if individual_probability
        return probs;
    end

    target_conj = conj(target)
    if individual_fidelity
        @tullio fidelity[c,d] := target_conj[i] * rho_s[i,j,c,d] * target[i]
        return real(fidelity)
    end
    # takes an implicit sum and computes fidelity
    @tullio fidelity := target_conj[i] * rho_s[i,j,c,d] * target[j]
    return -real(fidelity) / s.anc_dim^(iter)
    # return -real(fidelity)
end


function rand_density(n=2)
    gin = rand(n, n)
    gin = gin + 1im * rand(n, n)
    rho = gin * gin'
    rho = rho / tr(rho)
    return rho
end

function rand_tree(s::System, iter)
    params = [rand_param(s) for _ in 1:total_tree_count(s.anc_dim, iter)];
    params = reduce(hcat, params)
    return params
end

function opt(s::System, target, params, noise_level)
    rand_states = [rand_density(s.dim) for _ in 1:2]

    loss(params) = sum(forward_pass.(Ref(s), Ref(target), Ref(params), rand_states, noise_level)) / size(rand_states,1)
    j_loss(params) = reshape(Zygote.jacobian((x) -> loss(x), params)[1], size(params))

    result = optimize(loss, j_loss, params, LBFGS(), Optim.Options(store_trace=true, iterations=200); inplace=false)
    sol = Optim.minimizer(result)
    trace = Optim.trace(result)
    loss_history = [parse(Float64, split(string(t))[2]) for t in trace]
    return sol, loss_history
end

function opt_vs_iter(s::System, target, iter, noise_level)
    params = MIQSback.rand_tree(s, iter)

    return opt(s, target, params, noise_level)
end



greet() = print("Hello World! XD")

include("visualization.jl")


end # module MIQSback



