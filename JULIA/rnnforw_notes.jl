# PLEASE NOTE you can't run the following julia function, unfortunately I am not planning to publish the related data. However, you may create your own random matrices and check with your expectations.

# Extremely ugly and silly rnnforw!!,
# Therefore we need some notes to better understand it!

# Suppose you are given an input:
#=
mx2 = KnetArray(randn(Float32, 2048, 17))

julia> print("bs1 = ");for item in bs1;print(item, " "); end; # very very ugly batchSizes!
bs1 = 2 2 2 2 2 2 2 1 1 1

julia> y_mx2, hy_mx2, = rnnforw(r, wr, KnetArray(mx2),hy=true,batchSizes=bs1);

What do you expect? hy_mx2 should give you all the time step's hiddens' right?
Well, the opposite happens in this case!!!!!, namely,
y_mx2 is of size 16 x 17

Now, let's diggin the details of rnnforw (what a function uuhh..)
Suppose, we extracted the first instance related columns from mx2 as i1 (based on batchSizes argument)

# This is how we extract the 2nd instance from the columns of the mx2
i2 = []; for i=1:length(bs1); item=mx1[i];(bs1[i]==1)&&continue;push!(i2, item[:,2]);end;

bs_i1 = map(x->1, 1:size(i1,2))

# rindcs corresponds to related columns of mx2
julia> for i in rindcs; print(i, " "); end;
1 3 5 7 9 11 13 15 16 17

isapprox(y_mx2[:,rindcs], y_i1) # returns true if you think you are right!
Long story short here is the related alignment
y_mx2=[h1_1 h2_1 h1_2 h2_2 ....] what an ugly representation!!!!!
=#
using JSON, MAT, Knet, Images, JLD
include("conv_model.jl")
include("preprocess.jl")
include("model1.jl")

function test_rnnforw()
    hiddim = 16; indim = 2048;
    all_data = minibatch1(;batchsize=2) # length(all_data) == # of minibatches
    mx1, bs1 = all_data[1]

    mx2 = KnetArray(hcat(mx1...)) # all data
    r, wr = rnninit(indim, hiddim)
    y_mx2, hy_mx2, = rnnforw(r, wr, mx2, batchSizes=bs1,hy=true)

    # Extract the first instance, remember test batchsize is 2.
    rindcs1 = collect(1:2:15); push!(rindcs1, 16); push!(rindcs1, 17);
    i1 = []; for item in mx1; push!(i1, item[:,1]);end; i1=KnetArray(hcat(i1...));
    bs_i1 = map(x->1, 1:size(i1,2))
    y_i1, hy_i1 = rnnforw(r, wr, i1, hy=true, batchSizes=bs_i1)
    
    # Extract the second instance
    rindcs2 = collect(2:2:14); 
    i2 = []; for i=1:length(bs1); item=mx1[i];(bs1[i]==1)&&continue;push!(i2, item[:,2]);end;i2=KnetArray(hcat(i2...))
    bs_i2 = map(x->1, 1:size(i2,2))
    y_i2, hy_i2 = rnnforw(r, wr, i2, hy=true, batchSizes=bs_i2)
    
    # if the following assertions pass, then above explanations are correct!
    @assert isapprox(y_mx2[:,rindcs1], y_i1); @assert isapprox(y_mx2[:,rindcs2], y_i2);
    println("Wollaaa tests passed!")
    (isinteractive() && return (y_mx2, y_i1, y_i2, rindcs1, rindcs2))
end

