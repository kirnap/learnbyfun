include("train.jl")

function predme(seqid, imds)
    resmodel = create_resmodel()
    # for debug purposes
    seqid = 31011385
    imds = [5,2,4,1]
    sort!(imds)
    bundle = JLD.load("600_all_best_model.jld")
    model, ss, se = bundle["model"], bundle["ss"], bundle["se"]
    inputs = propfmod(seqid, imds, resmodel)
    unshift!(inputs, ss); push!(inputs, se);
    inputs = map(x->reshape(x, length(x), 1), inputs)
    f = (gpu()>= 0 ? KnetArray : Array)
    in2lstm = map(x->wembed(model)*f(x), inputs)
    
    # give half of them to forward and the other to backward
    r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);
    mid = div(length(imds), 2)

    forwin = hcat(in2lstm[1:mid]...) # forw input
    backin = hcat(in2lstm[mid+1:end]...) # back input

    hy_forw, = rnnforw(r_f, wr_f, forwin, batchSizes=map(t->1, 1:mid))
    hy_back, = rnnforw(r_b, wr_b, backin, batchSizes = map(t->1, mid+1:length(in2lstm)))
    softin = reshape(vcat(hy_forw[:,end], hy_back[:,end]), 1200,1)
    scores = wsoft(model) * softin  .+ bsoft(model)
    indx = sortperm(vec(Array(scores)), rev=true)
    catvocab = create_catvocab("data/label/train_no_dup.json")
    return indx, catvocab
    
end
