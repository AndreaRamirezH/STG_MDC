### Fix parameters to avoid trivial curve evolution
#Only free parameters: gCaT, gA

include("calcium_neuron_mdc.jl")

#Load diff.eq. system
od,ic,_ , ps = Neuron_wHomeo_dynamics(t->0.);
tspan = (0,800.);
prob = ODEProblem(od,ic,tspan,ps)

free = ["gA_tgt", "gCaT_tgt"]
OF = only_free_params(last.(ps), get_name_ids(ps, free))
od, ic, ps = transform_problem(prob,OF; unames = first.(ic), pnames = first.(ps))
prob = ODEProblem(od,ic,tspan,ps)

#log transform problem
tstrct_log = logabs_transform(last.(ps))
od, ic, ps = transform_problem(prob, tstrct_log; unames=first.(ic), pnames = first.(ps))
nom_prob = ODEProblem(od,ic,tspan,ps);
p0 = last.(ps)

#solve transformed problem
nom_sol = solve(nom_prob, alg(),reltol=1e-11,abstol=1e-11)

grad_template=deepcopy(p0)
nom_ca_loss(p0,grad_template)
@show grad_template

#get initial direction
@time h = ForwardDiff.hessian(nom_ca_loss, p0)
@show δp0 = eigen(h).vectors[:,1]
# δp0 = [-0.6197238436539504, -0.7848199523500751]

init_dir = δp0
span = (0.,1.0)
momentum = 1000.;

curve_prob = specify_curve(nom_ca_loss, p0, init_dir, momentum, span)
cb = VerboseOutput(:low, 0.0:0.1:1.0)
@time mdc = evolve(curve_prob, alg,callback=cb)

P_all = plot(mdc)
savefig(P_all,"nom_ca_logfix_mdc.png")
cc = [nom_ca_loss(el) for el in eachcol(trajectory(mdc))];
p2 = plot(distances(mdc), log.(cc), ylabel = "log(cost)", xlabel = "distance", title = "cost over MD curve");
savefig(p2,"cost_over_logfixcurve.png")
