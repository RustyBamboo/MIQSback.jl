using Graphs
using GraphRecipes
using Plots

function get_history(s::System, target, params, noise_level, rho_s)
    iter = compute_tree_depth(s.anc_dim, size(params, 2)) + 1

    history = []
    push!(history, real(target' * rho_s * target))
    for i in 1:iter-1
        # display(params[:, 1:2^(i+1)-1])
        push!(history, reduce(vcat, forward_pass(s, target, params[:,1:total_tree_count(s.anc_dim,i)], rho_s, noise_level; individual_fidelity=true)))

    end
    return  history
end

function get_history_rho(s::System, target, params, noise_level, rho_s)
    iter = compute_tree_depth(s.anc_dim, size(params, 2)) + 1

    history = []
    push!(history, rho_s)
    for i in 1:iter-1
        # display(params[:, 1:2^(i+1)-1])
        display(1:total_tree_count(s.anc_dim,i+1))
        push!(history, reduce(vcat, forward_pass(s, target, params[:,1:total_tree_count(s.anc_dim,i)], rho_s, noise_level; individual_rho=true)))
    end
    return  history
end


function get_prob_history(s::System, target, params, noise_level, rho_s)
    iter = compute_tree_depth(s.anc_dim, size(params, 2)) + 1
    

    history = []
    push!(history, 1)

    for i in 1:iter-1
        # display(params[:, 1:2^(i+1)-1])
        push!(history, reduce(vcat, forward_pass(s, target, params[:,1:total_tree_count(s.anc_dim,i)], rho_s, noise_level; individual_probability=true)))

    end
    return  history
end

function plot_tree(fidelity, prob, iter=5)
    # Plotting tree
    tree = BinaryTree(iter + 1)

    
    function childs(i)
        return (2 * i), (2 * i) + 1
    end

    
    prob = Array(reduce(vcat, prob))
    println(length(prob))

    function label!(prob, dict, cum_prob, i=1)
        if i ≥ 2^iter 
            return 0
        end

        l, r = childs(i)

        cum_prob[l] = cum_prob[i] * prob[l]
        cum_prob[r] = cum_prob[i] * prob[r]

        pr_0 = round(Int, prob[l] * 100)
        pr_1 = round(Int, prob[r] * 100)

        dict[(i, l)] = "0 ($(pr_0)%)"
        dict[(i, r)] = "1 ($(pr_1)%)"
        # dict[(i,l)] = @sprintf("0 (%.2f)", pr_0)
        # dict[(i,r)] = @sprintf("1 (%.2f)", pr_1)

        label!(prob, dict, cum_prob, l)
        label!(prob, dict, cum_prob, r)
    end
    d = Dict()
    cum_prob = zeros(Float64, size(prob))
    cum_prob[1] = 1
    label!(prob, d, cum_prob)

    println("diff")
    println(cum_prob)

    fidelity = Array(reduce(vcat, fidelity))

    colors = get(cgrad(:acton), fidelity)
    graph = graphplot(tree; curves=false, root=:left, curvature_scaler=0, method=:buchheim, nodesize=0.3, nodeshape=:circle, nodecolor=colors, edgelabel=d, )

    blank = plot(foreground_color_subplot=:white)
    # bar = scatter([0, 0], [1, 1], zcolor=[0, 1],
        # xlims=(1, 1), label="", c=:inferno, levels=15, colorbar_title="Fidelity", framestyle=:none)

    # bar = scatter([0], [0], zcolor=[0,1], label="", c=:viridis, colorbar_title="cbar", background_color_subplot=:transparent, markerstrokecolor=:transparent, framestyle=:none, inset=bbox(0.1, 0, 0.6, 0.9, :center, :right), subplot=1)
    
    l = @layout [a b{0.005w}]
    plot(graph, blank, layout=l; margin=0Plots.mm, left_margin=0Plots.mm)
    p_all = scatter!([0], [0], markeralpha=0, alpha=1, zcolor=[0,1], label="", c=:acton, colorbar_title="Fidelity", background_color_subplot=:transparent, markerstrokecolor=:transparent, framestyle=:none,
    inset=bbox(0.05, 0.1, 0.6, 0.75, :top, :right), subplot=3)


    high = 2^(iter+1) - 1
    low = 2^iter


    texts = ["$(round(Int, x * 100))%" for x in reverse(cum_prob[low:high])]
    bar!(1:(high-low+1),
        reverse(cum_prob[low:high]), 
        inset = (1, bbox(0.79, 0.083, 0.1, 0.835)),
        # inset = (1, bbox(0.79, 0.083, 0.1, 0.835)),
        # inset = (1, bbox(0.79, 0.06, 0.1, 0.88)),
        orientation=:horizontal,
        subplot=4,
        bg_inside = nothing,
        label=false,
        legend=false,
        ticks=false,
        showaxis=false,
        bar_width = 0.3,
        fillcolor=:cornflowerblue,
        # series_annotations = text.(texts, halign=:left, valign=:vcenter, 7),
        )
    return p_all
end
