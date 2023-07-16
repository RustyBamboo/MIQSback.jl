using MIQSback
using Plots
using LinearAlgebra

system = MIQSback.System(2,2)

iter = 3

target = [1.0+0im 1]' / sqrt(2)
sol, loss_history =  MIQSback.opt_vs_iter(system, target, iter, 0)

plot(loss_history, label="loss")
xlabel!("Optimization")
ylabel!("-Fidelity")

rho_i = [3.0 + 0im; -10.1]
rho_i = rho_i / norm(rho_i)
rho_i = rho_i * rho_i'

fidelity = MIQSback.get_history(system, target, sol, 0, rho_i)
prob = MIQSback.get_prob_history(system, target, sol, 0, rho_i)

MIQSback.plot_tree(fidelity, prob, iter)
