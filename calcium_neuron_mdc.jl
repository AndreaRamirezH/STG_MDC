using OrdinaryDiffEq, ModelingToolkit, MinimallyDisruptiveCurves, Plots, DiffEqParamEstim, DiffEqSensitivity
using NumericalIntegration, LinearAlgebra, ForwardDiff, FiniteDiff

const D_noise = 0.01
const VNa = 50
const VK = -80
const VCa = 100
const VCl = -20
const Vleak = -5
const taug = 1.0e3
const tauNa = 1.0e3
const Catgt = 50.
const gleak = 0.01

boltz(V,A,B)= 1.0 ./(1.0 + exp((V+A)./B))

tauX(V,A,B,D,E)= A - (B*boltz(V,D,E))

function Neuron_wHomeo_dynamics(input)

    @parameters t
    D = Differential(t)

    @parameters  gNa_tgt gKd_tgt gA_tgt gCaT_tgt gCaS_tgt gKCa_tgt gH_tgt
    paramvars = [gNa_tgt, gKd_tgt, gA_tgt, gCaT_tgt, gCaS_tgt, gKCa_tgt, gH_tgt]

    @variables V(t) Ca(t) mNa(t) hNa(t) mKd(t) mA(t) hA(t) mCaT(t) hCaT(t) mCaS(t) hCaS(t) mKCa(t) mH(t)
    @variables gNa(t) gKd(t) gA(t) gCaT(t) gCaS(t) gKCa(t) gH(t) RNa(t) RKd(t) RA(t) RCaT(t) RCaS(t) RKCa(t) RH(t)
    statevars = [V, Ca, mNa, hNa, mKd, mA, hA, mCaT, hCaT, mCaS, hCaS, mKCa, mH,
                gNa, gKd, gA, gCaT, gCaS, gKCa, gH, RNa, RKd, RA, RCaT, RCaS, RKCa, RH]

    #gene expression taus
    tauCaT = (gNa_tgt/gCaT_tgt)*tauNa
    tauCaS = (gNa_tgt/gCaS_tgt)*tauNa
    tauA = (gNa_tgt/gA_tgt)*tauNa
    tauKd = (gNa_tgt/gKd_tgt)*tauNa
    tauKCa = (gNa_tgt/gKCa_tgt)*tauNa
    tauH = (gNa_tgt/gH_tgt)*tauNa

    #currents
    INa = gNa*mNa^3*hNa*(VNa - V)
    ICaS= gCaS*mCaS^3*hCaS*(VCa - V)
    ICaT= gCaT*mCaT^3*hCaT*(VCa - V)
    Ih= gH*mH*(VCl - V)
    IKa= gA*mA^3*hA*(VK - V)
    IKCa= gKCa*mKCa^4*(VK - V)
    IKd= gKd*mKd^4*(VK - V)
    Ileak= gleak*(Vleak - V)


    #gating
    mNainf  = boltz(V,25.5,-5.29)
    taumNa  = tauX(V,1.32,1.26,120.,-25.)
    hNainf  = boltz(V,48.9,5.18)
    tauhNa  = (0.67/(1+exp((V+62.9)/-10.0)))*(1.5 + 1/(1+exp((V+34.9)/3.6)))

    mCaTinf  = boltz(V,27.1,-7.2)
    taumCaT  = tauX(V,21.7,21.3,68.1,-20.5)
    hCaTinf  = boltz(V,32.1,5.5)
    tauhCaT  = tauX(V,105.,89.8,55.,-16.9)

    mCaSinf  = boltz(V,33.,-8.1)
    taumCaS  = (1.4 + (7/((exp((V+27)/10))+(exp((V+70)/-13)))))
    hCaSinf  = boltz(V,60.,6.2)
    tauhCaS  = 60 + (150/((exp((V+55)/9))+(exp((V+65)/-16))))

    mAinf  = boltz(V,27.2,-8.7)
    taumA  = tauX(V,11.6,10.4,32.9,-15.2)
    hAinf  = boltz(V,56.9,4.9)
    tauhA  = tauX(V,38.6,29.2,38.9,-26.5)

    mKCainf = (Ca/(Ca+3.0))/(1+exp((V+28.3)/-12.6))
    taumKCa  = tauX(V,90.3,75.1,46.,-22.7)

    mKdinf  = boltz(V,12.3,-11.8)
    taumKd  = tauX(V,7.2,6.4,28.3,-19.2)

    mHinf  = boltz(V,70.,6.)
    taumH  = tauX(V,272.,-1499.,42.2,-8.73)

    Ca_inf = 0.05 + 0.94*(ICaS + ICaT)

    eqs = [
        D(V) ~          INa+ICaT+ICaS+IKa+IKCa+IKd+Ih+Ileak,
        D(Ca) ~                  (1/20.)*(Ca_inf - Ca), # originally no tauCa = 20
        D(mNa) ~                 (1/taumNa)*(mNainf - mNa),
        D(hNa) ~                 (1/tauhNa)*(hNainf - hNa),
        D(mCaS) ~                (1/taumCaS)*(mCaSinf - mCaS),
        D(hCaS) ~                (1/tauhCaS)*(hCaSinf - hCaS),
        D(mCaT) ~                (1/taumCaT)*(mCaTinf - mCaT),
        D(hCaT) ~                (1/tauhCaT)*(hCaTinf - hCaT),
        D(mH) ~                  (1/taumH)*(mHinf - mH),
        D(mA) ~                  (1/taumA)*(mAinf - mA),
        D(hA) ~                  (1/tauhA)*(hAinf - hA),
        D(mKCa) ~                (1/taumKCa)*(mKCainf - mKCa),
        D(mKd) ~                 (1/taumKd)*(mKdinf - mKd),

        D(gNa) ~                 (1/taug)*(RNa-gNa),
        D(gCaS) ~                (1/taug)*(RCaS-gCaS),
        D(gCaT) ~                (1/taug)*(RCaT-gCaT),
        D(gA) ~                  (1/taug)*(RA-gA),
        D(gKd) ~                 (1/taug)*(RKd-gKd),
        D(gKCa) ~                (1/taug)*(RKCa-gKCa),
        D(gH) ~                  (1/taug)*(RH-gH),
        D(RNa) ~                 (1/tauNa)*(Catgt-Ca),
        D(RCaS) ~                (1/tauCaS)*(Catgt-Ca),
        D(RCaT) ~                (1/tauCaT)*(Catgt-Ca),
        D(RA) ~                  (1/tauA)*(Catgt-Ca),
        D(RKd) ~                 (1/tauKd)*(Catgt-Ca),
        D(RKCa) ~                (1/tauKCa)*(Catgt-Ca),
        D(RH) ~                  (1/tauH)*(Catgt-Ca)

    ]

    ps = paramvars .=> [700.,90.,85.,6.25,2.25,50.,0.3]
    ics = statevars .=> [-60,0.05,0,0,0,0,0,0,0,0,0,0,0,780.,100.,94.,7.,2.5,55.,0.03,
                        780.,100.,94.,7.,2.5,55.,0.03]

    od = ODESystem(eqs, t, statevars, paramvars)
    tspan = (0.,800.)
    return od, ics, tspan, ps
end

alg = Rodas5
solve = OrdinaryDiffEq.solve

# Cost function

t0 = 400.; tf = 800.

integrand(el1, el2) = sum(abs2, el1 - el2)

function redo(p,nom_prob)
    prob = remake(nom_prob; p=p)
    sol = solve(prob, alg())
    return prob,sol
end

function loss2(p,nom_prob,nom_ca)
  prob,sol = redo(p,nom_prob)
  new_t0_idx = findfirst(t-> t>t0, sol.t)
  ca_sol = sol[2,new_t0_idx:end]
  m = min(length(ca_sol),length(nom_ca)) #take same size ca trace in sol and nom_sol
  x = sol.t[end-m:end]
  y = abs2.(nom_sol[2,end-m:end] .- sol[2,end-m:end])
  loss2 = NumericalIntegration.integrate(x,y)
  return loss2
end

loss(p)=loss2(p,nom_prob,nom_ca)

function lossgrad(p,g)
  g[:] = ForwardDiff.gradient(p) do p
    loss(p)
  end
  return loss(p)
end
