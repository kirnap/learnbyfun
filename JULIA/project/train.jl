using Knet
include("model.jl")
include("data.jl")

opts = Dict{}()
ENEMBED = 8; get!(opts, :ENEMBED, ENEMBED)
DEMBED = 16; get!(opts, :DEMBED, DEMBED);
HIDDEN = 256; get!(opts, :HIDDEN, HIDDEN); 
BATCHSIZE = 16; get!(opts, :BATCHSIZE, BATCHSIZE);
ATTENTION = 100; get!(opts, :ATTENTION,  ATTENTION);
EPOCH = 10 ; get!(opts, :EPOCH, EPOCH);



realdat = "data/cmudict.dict" # data directory
function main()
    all_data = createData(realdat)
    data, envocab, devocab = all_data
    mbs = mb2(data, BATCHSIZE)
    println("opts=",[(k,v) for (k,v) in opts]...)
    model = initmodel(length(envocab), length(devocab), ENEMBED, DEMBED, HIDDEN, ATTENTION)
    optim = optimizers(model,Adam)
    m_test = mbs[1]; encd, decd = m_test; decx, _ = decd;
    a = accuracy(predict(model, m_test), decx)
    println("acc $a")
    for i in 1:EPOCH
        ltot = 0
        for m in mbs
            grads, lval = lossgradient(model, m)
            update!(model, grads, optim)
            ltot += lval
            #@show lval
        end
        a = accuracy(predict(model, m_test), decx)
        lav = ltot/length(mbs)
        println("EPOCH $i loss $lav acc.. $a")
    end        
end
main()
