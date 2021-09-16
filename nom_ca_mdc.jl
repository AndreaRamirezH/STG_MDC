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
nom_sol = solve(nom_prob, alg(),reltol=1e-11,abstol=1e-11)

#grad_template=deepcopy(p0)
#nom_ca_loss(p0,grad_template)
#@show grad_template

#get initial direction
#@time H = ForwardDiff.hessian(nom_ca_loss, p0)
#@show δp0 = eigen(H).vectors[:,1]

#evolve curve
init_dir = [0.37796447298540337, 0.37796447304046404, 0.3779644729805194, 0.3779644729682884, 0.37796447305048764, 0.3779644730511052, 0.377964472988322]
span = (0.,1.0)
momentum = 500.;

curve_prob = specify_curve(nom_ca_loss, p0, init_dir, momentum, span)
cb = VerboseOutput(:low, 0.1:0.2:1.0)
@time mdc = evolve(curve_prob, alg,callback=cb)

P_all = plot(mdc;idxs = collect(1:7))

cc = [nom_ca_loss(el) for el in eachcol(trajectory(mdc))];
p2 = plot(distances(mdc), log.(cc), ylabel = "log(cost)", xlabel = "distance", title = "cost over MD curve");
savefig(p2,"cost_over_curve_to1.png")
