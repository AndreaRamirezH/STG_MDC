include("calcium_neuron_mdc.jl")

#Load diff.eq. system
od,ic,_ , ps = Neuron_wHomeo_dynamics(t->0.);
tspan = (0,800.);
prob = ODEProblem(od,ic,tspan,ps)

#log transform problem
tstrct_log = logabs_transform(last.(ps))
od, ic, ps = transform_problem(prob, tstrct_log; unames=first.(ic), pnames = first.(ps))
nom_prob = ODEProblem(od,ic,tspan,ps);
p0 = last.(ps)

#solve transformed problem
nom_sol = solve(nom_prob, alg())
tsteps_from_t0 = filter(t-> t>t0, nom_sol.t)
t0_idx = findfirst(t-> t>t0, nom_sol.t)
nom_ca = nom_sol[2,t0_idx:end];

#create cost with 2 methods
@show loss(p0)

grad_template = deepcopy(p0)
@show lossgrad(p0,grad_template)

cost_created = DiffCost(loss, lossgrad);
@show cost_created(p0)


@time H = ForwardDiff.hessian(loss, p0)

#evolve curve
init_dir = [δp0[1],0.0,δp0[3],δp0[4],δp0[5],δp0[6],δp0[7]];
span = (-15.,15.)
momentum = 5000.;

curve_prob = curveProblem(cost_created, p0, init_dir, momentum, span)
@time mdc = evolve(curve_prob, Rodas5)
