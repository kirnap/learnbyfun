include("train.jl")

function predme()
    resmodel = create_resmodel()
    seqid = 31011385
    imds = [5,2,4,1]
    sort!(imds)
    bundle = JLD.load("600_all_best_model.jld")
    model, ss, se = bundle["model"], bundle["ss"], bundle["se"]
    inputs = propfmod(seqid, imds, resmodel)

    inputs = map(x->reshape(x, length(x), 1), inputs)
    f = (gpu()>= 0 ? KnetArray : Array)
    in2lstm = map(x->wembed(model)*f(x), inputs)
    
    # give half of them to forward and the other to backward
    r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);
    
end
