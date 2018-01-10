using Knet
include("model.jl")
include("data.jl")

ENEMBED = 8
DEMBED = 16
HIDDEN = 256
BATCHSIZE = 64
ATTENTION = 100
EPOCH = 3

realdat = "data/cmudict.dict" # data directory
function main()
    all_data = createData(realdat)
    data, envocab, devocab = all_data
    mbs = mb2(data, BATCHSIZE)

    model = initmodel(length(envocab), length(devocab), ENEMBED, DEMBED, HIDDEN, ATTENTION)
    optim = optimizers(model,Adam)
    return nothing
    for i in 1:EPOCH
        ltot = 0
        for m in mbs
            grads, lval = lossgradient(model, m)
            update!(model, grads, optim)
            @show lval
            ltot += lval
            
        end
        println("EPOCH $i loss $ltot")
    end        
end
main()
