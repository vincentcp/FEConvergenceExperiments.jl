
module FEConvergenceExperiments
using BasisFunctions, FrameFun, DomainSets, QuadGK, JLD, SpecialFunctions, DelimitedFiles, IJulia, PGFPlots, ApproxFun, LaTeXStrings
thispath = splitdir(@__FILE__)[1]
ELT = BigFloat; prec = 512;
setprecision(prec);
# Random interior point
r = ELT(.29384)
# Right hand side
a = ELT(1//2)
# maximum frequency
K = 64
# M number of Laguere quadrature points
M = 5000
# tolerance of quadgk
tol = max(1e-60, sqrt(eps(real(ELT))))

# Special functions to approximate
function spline_function(p::Int, α::T, x::T) where T<:Real
    if x < r
        1-(-x/r+1)^p
    else
        1-((x-r)/(α-r))^p
    end
end
jump_function(b,x::T) where T<:Real = x<b ? x : T(1)
cusp = x->Complex(abs(x-r^ELT(1//4)))
dini = x->Complex(ELT(log((1//2)*abs(x-r)))^ELT(-2))
H3 = x->spline_function(3, a, x)
H9 = x->spline_function(9, a, x)
H15 = x->spline_function(15, a, x)
# Generate Laguerre Quadrature points (used in Steepest descent)
if isfile("/../data/quad_numbersK$(prec)M$(M).jld")
    println("Load quadrature")
    laguerre_x0, laguerre_w0, laguerre_x, laguerre_w = load(thispath*"/../data/quad_numbersK$(prec)M$(M).jld","laguerre_x0","laguerre_w0", "laguerre_x", "laguerre_w")
    @info("Quadrature Loaded")
else
    @info "Creating quadrature"
    using GaussQuadrature
    laguerre_x0, laguerre_w0 = laguerre(M, real(ELT)(0))
    laguerre_x, laguerre_w = laguerre(M+20, real(ELT)(0))
    JLD.save(thispath*"/../data/quad_numbersK$(prec)M$(M).jld", "laguerre_x0",laguerre_x0, "laguerre_w0",laguerre_w0,"laguerre_x",laguerre_x,"laguerre_w",laguerre_w)
end
# Right hand sides

# x^α integration
"Quadrature, most general approach."
function quadgk_rhs(k::Int, a, f::Function, ELT)
    @assert typeof(a) == real(ELT)
    quadgk(x->f(x)*exp(-1im*2ELT(pi)*k*x),real(ELT(0)),real(a/2),real(a),rtol=tol,atol=tol)[1]
end
"∫_0^a exp(-1im2pikx)*x^α dx using steepest descent."
function frths_int(k::Int, a, α, ELT, prec)
    @assert typeof(a) == real(ELT)
    omega = 2ELT(pi)*k
    if k != 0
        (-ELT(1)*1im/omega)^ELT(α+1)*gamma(real(ELT)(α+1))+1im*exp(-1im*omega*a)/omega*gl(omega, a, ELT(α), prec)
    else
        ELT(1/(α+1))*a^(α+1)+0im
    end
end
"Gauss Laguerre quadrature with a fixed number of points."
function gl(omega::ELT, a, α, prec) where ELT
    @assert real(typeof(a)) == real(typeof(α)) == real(ELT)
    r0 = sum(laguerre_w0.*(a .- 1im*laguerre_x0/omega).^ELT(α))
    r = sum(laguerre_w.*(a .- 1im*laguerre_x/omega).^ELT(α))
    if abs(r-r0) > 1e3eps(real(ELT))
        @show abs(r-r0) , 1e3eps(real(ELT))
    end
    r
end
cusp_rhs = (k::Int,ELT,prec) -> Complex(exp(-2im*ELT(pi)*k*r)*(frths_int(-k, r,ELT(1//4), ELT, prec) + frths_int(+k, a-r,ELT(1//4), ELT, prec)  ))

# Spline integration
function spline_integral(k::Int, p::Int, α)
    T = eltype(α)
    pi = convert(T,π)
    if k==0
        f = α
    else
        f = (exp(1im*2T(pi)*k*α)-1)/(1im*2*T(pi)*k)
    end
    f - exp(2im*T(pi)*k*r)*((-r)^(-p)*monomial_integral(k, p, -r, convert(T, 0)) +
                                   (α-r)^(-p)*monomial_integral(k, p, convert(T, 0), α-r))
end
function monomial_primitive(k::Int, p::Int, x::ELT) where ELT
    k==0 ? x^ELT(p+1)/(p+1) : exp(1im*2ELT(pi)*k*x)*sum([ ELT(-1)^ELT(l)*ELT(factorial(p))/ELT(factorial(p-l))/(1im*2ELT(pi)*k)^ELT(l+1)*x^ELT(p-l)  for l in 0:p])
end
monomial_integral(k::Int, p::Int, a::T, b::T) where T<:Real = monomial_primitive(k, p, b)-monomial_primitive(k, p, a)

jump_rhs(k, a, b, ::Type{ELT}) where ELT = monomial_integral(-k, 1, real(ELT)(0), real(ELT)(b)) + (k==0 ? ELT(a)-ELT(b) : (exp(-2im*ELT(pi)*k*a)-exp(-2im*ELT(pi)*k*b))/(-2im*ELT(pi)*k))


# Combining functions with their rhs
funs = Dict(
    "Holder conv" => (x->x^ELT(3//4), (k->frths_int(k, a, convert(ELT,3//4), ELT, prec))),
    "Holder div" => (x->x^ELT(1//10), (k->frths_int(k, a, convert(ELT,1//10), ELT, prec))),
    "Holder lim" => (x->x^ELT(1//2), k->frths_int(k, a, convert(ELT,1//2), ELT, prec)),
    "Jump" => (x->jump_function(a/2,x), (k->jump_rhs(k, a, convert(ELT,a/2), ELT))),
    "Cusp" => (cusp, (k->cusp_rhs(k,ELT, prec))),
    "Dini" => (dini, k->quadgk_rhs(k, a, dini, ELT)),
    "H3" => (H3, (k->spline_integral(-k, 3, a) )),
    "H9" => (H9, (k->spline_integral(-k, 9, a))),
    "H15" => (H15, (k->spline_integral(-k, 15, a)  )))


function BasisFunctions.Gram(src::Dictionary; options...)
    T = codomaintype(src)
    A = zeros(T,length(src),length(src))*NaN
    BasisFunctions.gram_matrix!(A, src)
    MatrixOperator(src, src, A)
end

my_prec(ELT) = abs(round(Int,log10(eps(ELT))))

BASIS = FourierBasis{ELT}
dom = Interval(ELT(0),a)
B = BASIS(7)
F = extensionframe(B, dom)
k = keys(funs)
v = [funs[ki] for ki in k]
fs = ntuple(k->v[k][1],length(v))
rhss = ntuple(k->v[k][2],length(v))


rhs(B::ExtensionFrame) =
    [Complex(rhsi(convert(Int,j))) for j in ordering(B), rhsi in rhss]

function create_data()
    setprecision(prec)
    Ns = [(1<<k)+1 for k in 1:convert(Int,log2(K))]
    errorl = zeros(Float64, length(Ns), length(k))
    errorr = similar(errorl)
    errori = similar(errorl)
    errorm = similar(errorl)
    accuracy = similar(errorl)
    x = nothing
    S = nothing
    for (i,N) in enumerate(Ns)
        println()
        @show N
        B = BASIS(N)
        F = extensionframe(B, dom)
        G = Gram(F)
        b = rhs(F)
        x = matrix(G)\b
        @show norm(x)
        S = [Expansion(F, x[:,i]) for i in 1:size(x,2)]
        xl = ELT(0)
        xi = r
        xr = a
        xm = a/2
        errorl[i,:] .=  [abs(f(xl)-sum(x[:,i])) for (f,i) in zip(fs,1:size(x,2))]
        errori[i,:] .=  [abs(f(xi)-Si(xi)) for (f,Si) in zip(fs,S)]
        errorm[i,:] .=  [abs(f(xm)-Si(xm)) for (f,Si) in zip(fs,S)]
        errorr[i,:] .=  [abs(f(xr)-sum(x[:,i].*[(-1)^convert(Int,k) for k in ordering(F)])) for (f,i) in zip(fs, 1:size(x,2))]

        writedlm(thispath*"/../data/left_error$(my_prec(ELT))", errorl)
        writedlm(thispath*"/../data/right_error$(my_prec(ELT))", errorr)
        writedlm(thispath*"/../data/interior_error$(my_prec(ELT))", errori)
        writedlm(thispath*"/../data/accuracy$(my_prec(ELT))", accuracy)
        writedlm(thispath*"/../data/mid_error$(my_prec(ELT))", errorm)

        @show k
        @show errorl[i,:]
        @show errorr[i,:]
        @show errorm[i,:]
        @show errori[i,:]
    end
end

plot_data() = IJulia.notebook(;dir=thispath*"/../")
end
