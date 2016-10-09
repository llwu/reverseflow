
import Base.Test

immutable Interval
  a
  b
end

immutable ParametricInverse
  inverse_of::Function
  inv_f::Function
  inp_type::Type
  param_space::Type
end

gen(::Type{Real}) = rand()
gen(::Type{Integer}) = rand(-1000:1000)
gen(::Type{Interval}) = rand(-1000:1000)

function test_pinv(pinv)
  rand_params = gen(pinv.param_space)
  rand_inp = gen(pinv.inp_type)
  inv_output = pinv.inv_f(rand_inp..., rand_params...)
  fwd_output = pinv.inverse_of(inv_output...)
  fwd_output, rand_inp
end

ntests = 1000
inv_add(z, Θ) = (Θ, z-Θ)
inv_add_pi = ParametricInverse(+, inv_add, Real, Real)

inv_sub = ParametricInverse(-, (z, Θ)->(Θ+z, Θ), Real, Real)
inv_mul = ParametricInverse(*, (z, Θ)->(z^Θ, z^(1-Θ)), Real, Real)
inv_div = ParametricInverse(/, (z, Θ)->(z^Θ, z^(Θ-1)), Real, Real)

inv_pow = ParametricInverse(^, (z, Θ)->(Θ, log(Θ, z)), Real, Real)
inv_log = ParametricInverse(log, (z, Θ)->(Θ, Θ^z), Real, Real)

inv_sin = ParametricInverse(asin, (z, Θ)->(asin(z)+Θ*pi,), Real, Integer)

parametric_inverses = [inv_add_pi, inv_sub, inv_mul, inv_div, inv_pow, inv_log,
                       inv_sin]

for i in 1:ntests
  for pi in parametric_inverses
    ffy = ≈(test_pinv(pi)...)
    if !ffy
      println(pi)
    end
    @test ffy
  end
end
