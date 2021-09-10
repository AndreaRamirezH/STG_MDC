include("calcium_neuron_mdc.jl")

#Load diff.eq. system
od,ic,_ , ps = Neuron_wHomeo_dynamics(t->0.);
tspan = (0,800.);
nom_prob = ODEProblem(od,ic,tspan,ps)
p0 = last.(ps)

#solve problem
nom_sol = solve(nom_prob, alg(),reltol=1e-8,abstol=1e-8)

#create cost with 2 methods
@show loss(p0)

grad_template = deepcopy(p0)
@show lossgrad(p0,grad_template)

cost_created = DiffCost(loss, lossgrad);
@show cost_created(p0)

#@time H = ForwardDiff.hessian(loss, p0)
#@show δp0 = eigen(H).vectors[:,1]
δp0 = (eigen(H)).vectors[:, 1] = [-0.9822150772924748, -0.12696156057171237, -0.11895658261119682, -0.008771163326679782, -0.0031522783682580847, -0.06997562810953281, -0.0004205183238889107]

#evolve curve
init_dir = δp0;
span = (-10.,10.)
momentum = 5000.;

curve_prob = curveProblem(cost_created, p0, init_dir, momentum, span)
@time mdc = evolve(curve_prob, Rosenbrock23)
