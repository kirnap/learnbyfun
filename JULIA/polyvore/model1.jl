# First model

function initx(d...; ftype=Float32) # xavier
    if gpu() >=0
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end
end


function initz(d...; ftype=Float32) # zero
    if gpu() >=0
        KnetArray{ftype}(zeros(d...))
    else
        Array{ftype}(zeros(d...))
    end
end


function initmodel(embspace, hiddens, outdim, indim)
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
    model = Any[]
    push!(model, initx(hiddens, indim)) # special embeddings

    push!(model, initx(outdim, hiddens)) # MLP weight
    push!(model, initx(outdim, 1)) # MLP bias
end

# You need to concat the inputs and the query image and give the logistic loss
# Therefore you will learn multiple combinations


function minibatch1(;batchsize=32, tdata=nothing)
    # start and end tokens
    if tdata == nothing
        tdata = JLD.load("data/train_data.jld")
        tdata = filter(x->length(x)>2, tdata["dataset"])
    end
    ret = Any[];
    ss = randn(Float32, 2048); se = randn(Float32, 2048);i2=length(tdata)
    for i=1:batchsize:i2
        j = min(i2, i+batchsize-1)
        sij = sort(tdata[i:j],by=length, rev=true) # mb those sents
        T = length(sij[1]) # sorted-longest
        mx = []; bs = Int[]; my = [];
        for t in 0:(T+1)
            b = 0; xs = []
            for s in sij # s is the whole sequence
                if t == 0
                    push!(xs, ss)
                    b += 1
                elseif t<= length(s)
                    push!(xs, s[t].vivec)
                    b += 1
                elseif t == length(s)+1 # seq consumed
                    push!(xs, se)
                    b += 1
                else
                    break
                end
            end
            push!(mx, hcat(xs...)); push!(bs, b);
        end
        push!(ret, (mx, bs))
    end
    return ret
end


function minibatch2(;batchsize=32, tdata=nothing, ncols=nothing, ss=nothing, se=nothing)
    if tdata == nothing
        tdata = JLD.load("data/train_data.jld")
        tdata = filter(x->length(x)>2, tdata["dataset"])
    end
    ret = Any[];
    if ss==nothing
        ss = randn(Float32, 2048); se = randn(Float32, 2048);
    end
    find_cids(x) = map(t->t.cid, x);i2=length(tdata);
    for i=1:batchsize:i2
        j = min(i2, i+batchsize-1)
        sij   = sort(tdata[i:j],by=length, rev=true) # mb those sents
        sij_b = map(reverse, sij)
        T = length(sij[1]) # sorted-longest

        # get the normal order
        mx_f = []; bs_f = Int[]; my_f=[]
        for t in 0:T-1 # no need for the last
            b = 0; xs_f=[];
            for s in sij
                if t == 0
                    push!(xs_f, ss)
                    b += 1
                elseif t<length(s)
                    push!(xs_f, s[t].vivec)
                    b += 1
                else
                    break
                end
            end
            push!(mx_f, hcat(xs_f...)); push!(bs_f, b);
        end
        
        # get the reversed order
        mx_b = []; bs_b = Int[]; my_b = []
        for t in 0:T-1 # no need for the first
            b = 0; xs_b = [];
            for s in sij_b
                if t == 0
                    push!(xs_b, se)
                    b += 1
                elseif t<length(s)
                    push!(xs_b, s[t].vivec)
                    b += 1
                else
                    break
                end
            end
            push!(mx_b, hcat(xs_b...));push!(bs_b,b)
        end

        # sorry for not understandable coding
        # get the gold values for each instance, store linear indices
        ygolds = vcat(map(find_cids, sij)...);
        ygolds = map(i->ygolds[i]+(i-1)*ncols, 1:length(ygolds));

        state = ((hcat(mx_f...), bs_f),(hcat(mx_b...), bs_b), ygolds)
        push!(ret, state)
    end
    return ret, ss, se 
end


# accessors 
wembed(model) = model[1]
flstm(model)  = model[2]
blstm(model)  = model[3]
wsoft(model)  = model[4]
bsoft(model)  = model[5]
function initmodel(inputdim, embed, hidden, outdim)
    model = Any[]
    push!(model, initx(embed, inputdim))

    r_f, wr_f = rnninit(embed, hidden) # forward lstm [1]
    push!(model, (r_f, wr_f));

    r_b, wr_b = rnninit(embed, hidden) # backward lstm [2]
    push!(model, (r_b, wr_b));

    push!(model, initx(outdim, 2hidden))# soft layer 
    push!(model, initz(outdim, 1)) # bias

    return model
end


# context based bi-lstm loss
function biloss(model, mdata; dropout=(0.0, 0.0))
    (mx_f, bs_f), (mx_b, bs_b), ygolds = mdata
    r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);

    inforw = wembed(model)*(gpu()>=0 ? KnetArray(mx_f) : mx_f)
    inback = wembed(model)*(gpu()>=0 ? KnetArray(mx_b) : mx_b)
    hforws, = rnnforw(r_f, wr_f, inforw, batchSizes=bs_f)
    hbacks, = rnnforw(r_b, wr_b, inback, batchSizes=bs_b)
    softin  = vcat(hforws, hbacks[:, end:-1:1]) # reverse the backward lstm
    
    scores = wsoft(model) * softin .+ bsoft(model)
    logprobs  = logp(scores, 1)
    total = -logprobs[ygolds] / length(ygolds)
    return sum(total)
end

# ugly-minibatching
function biloss2(model, mdatas; pdrop=(0.0, 0.0))
    total =0.0
    for mdata in mdatas
        (mx_f, bs_f), (mx_b, bs_b), ygolds = mdata
        r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);

        inforw = wembed(model)*(gpu()>=0 ? KnetArray(mx_f) : mx_f)
        inback = wembed(model)*(gpu()>=0 ? KnetArray(mx_b) : mx_b)
        hforws, = rnnforw(r_f, wr_f, inforw, batchSizes=bs_f)
        hbacks, = rnnforw(r_b, wr_b, inback, batchSizes=bs_b)
        softin  = vcat(hforws, hbacks[:, end:-1:1]) # reverse the backward lstm
        
        scores = wsoft(model) * softin .+ bsoft(model)
        logprobs  = logp(scores, 1)
        tot = sum(-logprobs[ygolds]); 
        total += tot / length(ygolds)
    end
    return total / length(mdatas)
end


bigrad = gradloss(biloss)
bigrad2 = gradloss(biloss2)


function accuracy(model, alldata)
    ntot=0; ncorrect = 0
    for mdata in alldata
        (mx_f, bs_f), (mx_b, bs_b), ygolds = mdata
        r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);
        inforw = wembed(model)*(gpu()>=0 ? KnetArray(mx_f) : mx_f)
        inback = wembed(model)*(gpu()>=0 ? KnetArray(mx_b) : mx_b)
        hforws, = rnnforw(r_f, wr_f, inforw, batchSizes=bs_f)
        hbacks, = rnnforw(r_b, wr_b, inback, batchSizes=bs_b)
        softin  = vcat(hforws, hbacks[:, end:-1:1]) # reverse the backward lstm
        scores = wsoft(model) * softin .+ bsoft(model)
        logprobs  = scores#logp(scores, 1)
        ypreds = reshape(findmax(Array(logprobs),1)[2], length(ygolds))
        ncorrect += sum(ypreds .== ygolds)
        ntot +=  length(ygolds)
    end
    return ncorrect/ntot
end


function predict(model, instance)
    (mx_f, bs_f), (mx_b, bs_b), ygolds = mdata
    r_f, wr_f = flstm(model); r_b, wr_b = blstm(model);
    inforw = wembed(model)*(gpu()>=0 ? KnetArray(mx_f) : mx_f)
    inback = wembed(model)*(gpu()>=0 ? KnetArray(mx_b) : mx_b)
    hforws, = rnnforw(r_f, wr_f, inforw, batchSizes=bs_f)
    hbacks, = rnnforw(r_b, wr_b, inback, batchSizes=bs_b)
    softin  = vcat(hforws, hbacks[:, end:-1:1]) # reverse the backward lstm
    scores = wsoft(model) * softin .+ bsoft(model)
    logprobs  = vec(Array(scores))#logp(scores, 1)
    return sortperm(logprobs, rev=true)    
end


function propfmod(seqid, imds, remodel) # process for model
    # assume you are given cid and img ids
    seqpath = string("data/images/", seqid, "/")
    sort!(imds) # make sure that the ids are sorted
    inputs = Any[]
    for i in imds
        impath = joinpath(seqpath, string(i, ".jpg"))
        vivec  = get_imvec(impath, resmodel)
        push!(inputs, vivec)
    end
    return inputs
end


function create_catvocab(jsonfile)
    catvocab = Dict{}()
    bigdata = JSON.parsefile(jsonfile)
    for data in bigdata
        imgs = data["items"]
        (length(imgs) < 3 && continue); img_seq = [];
        for i in imgs
            ((i["name"] == "polyvore" || i["name"] == "") && continue)
            (length(img_seq) > 8 && break); # Todo: we may need more
            category = i["categoryid"]; cid = i["index"];
            push!(img_seq, 1) # actually no need but put for the rush
            if haskey(catvocab, category)
                if !(i["name"] in catvocab[category])
                    push!(catvocab[category], i["name"])
                end
            else
                catvocab[category] = Any[]; push!(catvocab[category], i["name"]);
            end
        end
    end
    return catvocab
end
