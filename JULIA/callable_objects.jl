# This script aims to discover 'callable objects in julia'

# holds the coefficients of a polynomial
# e.g. 100x^2 + 10x + 1 -> Polynomial([1,10,100])
struct Polynomial{T}
    coeffs::Vector{T}
end


# Used to calculate polynomial's value
function (p::Polynomial)(x) 
    v = p.coeffs[end]
    for k in 1:length(v)
        v = v*x + p.coeffs[k]
    end
    return v[end]
end

(p::Polynomial)() = p(5) # to define default caller

# In the function body p refers to the object that was called. You basically play with fields of a struct to generate desired results by 'calling an object'

#=
julia> p = Polynomial([1,10,100])
Polynomial{Int64}([1, 10, 100])

=#
