using Knet
import Statistics: mean

# TODO: copyto! for set methods

struct Multiply 
    w # weight
end


Multiply(;input::Int, output::Int, winit=xavier, o...) =  Multiply(param(output, input, init=winit, o...))

# methods
(m::Multiply)(x::Array{T}) where T<:Integer = m.w[:,x] # Enable Embeddings

# Relies on column oriented minibatching
(m::Multiply)(x) = m.w * x 


Embed = Multiply


struct BatchMul
    w # weight
end

BatchMul(;input::Int, output::Int, winit=xavier, o...) = BatchMul(param(output, input, init=winit), o...)

function (m::BatchMul)(x; flatten=false)
    xdims = size(x)
    x_flat = reshape(x, xdims[1], prod(xdims[2:end]))
    y = m.w * x_flat
    flatten ? y : reshape(y, size(y, 1), xdims[2:end]...)
end


struct Linear
    w::Multiply
    b # bias
end

function Linear(;input::Int, output::Int, winit=xavier, binit=zeros, o...)
    Linear(Multiply(input=input, output=output, o...), param(output, init=binit, o...))
end


(l::Linear)(x) = l.w(x) .+ l.b

struct Dense
    l::Linear
    f  # activation
end


function Dense(;input::Int, output::Int, activation=nothing, winit=xavier, binit=zeros, o...)

    if activation == nothing
        return Linear(input=input, output=output, winit=winit, binit=binit)
    end

    if typeof(activation) == Symbol
        activation = eval(activation)
    elseif typeof(activation) == String
        eval(Symbol(activatio))
    else
        # activation is a Knet function
    end

    return Dense(Linear(input=input, output=output, winit=winit, binit=binit, o...), activation)
end


(d::Dense)(x) = d.f.(d.l(x))


struct LSTM
    r::RNN
end


# Data-related
Lpad(p::Int,x::Array)=cat(p*ones(Int,size(x,1)),x;dims=2);leftpad  = Lpad;
Rpad(x::Array,p::Int)=cat(x,p*ones(Int,size(x,1));dims=2);rightpad = Rpad;

dataxy(x) = (x,Rpad(sort(x,dims=2),11)) # (x, ygold) pair


function main()

    # create-data
    B, maxL= 64, 15; # Batch size and maximum sequence length for training
    data = [dataxy([rand(1:9) for j=1:B,k=1:rand(1:maxL)]) for i=1:10000];
    
    hidden=3; embdim=9; vocab=11;
    emb = Embed(input=vocab, output=embdim); # input embedding
    encoder = RNN(embdim, hidden);
    decoder = RNN(embdim, hidden);

    # create data with minibatchsize 4 and seqlength of 2, size(x) = B,T -> 4,2
    x = [9 2
         4 1
         9 6
         7 2];
    x,y = dataxy(x)

    # Test your understanding
    ealls = encoder(emb(x)) # all outputs with an input size of (X, B, T) and outputsize (H, B, T)

    # Now suppose we are giving them 1-by-1
    # 1. Initialize hiddens, if empty zero initialization assumed
    hs = Any[]
    # 2. Go 1 step further in time
    e1 = encoder(emb(x[:, 1]), hidden=hs) # takes initially empty hidden, fills and returns

    # 3. Take the previous hidden, go 1 more step further
    e2 = encoder(emb(x[:, 2]), hidden=hs) # takes initially filled hidden, next state is affected

    # ENCODER
    
    # Tests
    @assert size(ealls) == (hidden, size(x)...)

    @assert ealls[:,:, 1] == e1 # first time step with zero hidden
    @assert ealls[:,:, 2] == e2 # second time step with first hidden initialized
    e2_wrong = encoder(emb(x[:,2])) # it is wrong because we assumed zero initialization for the second step
    @assert e2 != e2_wrong

    h_final = Any[]; ealls2 = encoder(emb(x), hidden=h_final)
    @assert h_final == hs
    @assert ealls2 == ealls

    
    # DECODER
    xdec = Lpad(10, x)
    dalls  = emb(xdec)
    @assert hs == h_final
    y_decode = decoder(dalls, hidden=h_final)
    # Once you give hidden kwarg, it modifies its hidden, following line proves that
    @assert hs != h_final

    # We flatten the RNN output within both time and minibatches
    s=size(y_decode); y_flat_time = reshape(y_decode, s[1], prod(s[2:end]));
    y_flat_time2 = hcat([y_decode[:,:,i] for i in 1:size(y_decode)[3]]...)
    @assert y_flat_time == y_flat_time2

    # Batchmul test
    Wout = Multiply(input=hidden, output=11);
    Bm = BatchMul(input=hidden, output=11); Bm.w.value = Wout.w.value;
    ypred1 = Wout(y_flat_time)
    ypred2 = Bm(y_decode, flatten=true)
    @assert ypred1 == ypred2

    # Loss calculation test
    lval = nll(ypred2, y) # y contains sorted value padded with eos
    ncol = vocab # number of class
    indx = [ y[i] + (i-1)*ncol for i in 1:length(y) ]
    @assert length(indx) == 4 * 3 # Here B is 4 T is 2 +1 for eos
    logprob = logp(ypred2, dims=1)
    lval2 = -mean(logprob[indx])
    @assert lval2 == lval
    @info "Tests passed!"    

    
    
    # y_decode contains all the hiddens for all the time steps
    return hs, h_final, dalls, decoder, y_decode
    
end



#= Quick tutorial

# You need to call J = @diff your_loss_funct;
# grad(J, anyparameter) will return your the gradient of the parameter
#
    function my_loss(model1, W, input)
         embs = W[:, input]
         hiddens = model1(embs)
         return sum(hiddens)
    end

input = rand(1:8, 4)
W = param(12, 8)
j = @diff my_loss(model1, W, input)
grad(J, anyparam) will give the gradient w.r.t. anyparam

# RNN
x = [9 2
     4 1
     9 6
     7 2];
x,y = dataxy(x)
inpu2rnn= emb(x)


# TODO: find a way to cache all the previous hiddens, check with matrix multiplication

# For matrix multiplication
# Suppose we are given an RNN output with the a size of (H, B, T), we want to reduce the dimension for classification
# What we need to do is to flatten out the matrix through the time dimension i.e. (H, BxT)
e.g.
julia> s = size(y_decode); y_flat_time = reshape(y_decode, s[1], prod(s[2:end]));
julia> @assert y_flat_time == hcat([y_decode[:,:,i] for i in 1:size(y_decode)[3]]...)

# If you want to see the whole columns you need to put following lines into the terminal
ENV["COLUMNS"]=120

=#
