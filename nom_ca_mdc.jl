include("calcium_neuron_mdc.jl")

#Load diff.eq. system
od,ic,_ , ps = Neuron_wHomeo_dynamics(t->0.);
tspan = (0,800.);
nom_prob = ODEProblem(od,ic,tspan,ps)
p0 = last.(ps)

#solve problem
nom_sol = solve(nom_prob, alg(),reltol=1e-8,abstol=1e-8)
tsteps = nom_sol.t
t0 = findfirst(t->t>=400.,nom_sol.t)

#create cost with 2 methods
integrand(el1, el2) = sum(abs2, sol[2,:] .- nom_sol[2,:])
lossf(sol, nom_sol) = sum( sum(abs2, sol[2,t0:end] .- nom_sol[2,t0:end])  )
lossf1(sol) = lossf(sol, nom_sol);

function mycost(p)
    return lossf1(redo(p,nom_prob,nom_sol)[2])
end
@show mycost(p0)

function mycost(p,grad)
    grad[:] = FiniteDiff.finite_difference_gradient( x -> mycost(x), p)
    #g = x -> ForwardDiff.gradient(mycost, x)
    #grad[:] = g(p)
    return mycost(p)
end

grad_template=deepcopy(p0)
mycost(p0,grad_template)
@show grad_template

#=
@time H = ForwardDiff.hessian(cost, p0)

@show δp0 = eigen(H).vectors[:,1]
δp0_1sttry = [-0.9822150772924748, -0.12696156057171237, -0.11895658261119682, -0.008771163326679782, -0.0031522783682580847, -0.06997562810953281, -0.0004205183238889107]
=#

#evolve curve
init_dir = δp0;
span = (0.,10.)
momentum = 1000.;

curve_prob = curveProblem(mycost, p0, init_dir, momentum, span)
@time mdc = evolve(curve_prob, Rosenbrock23)
