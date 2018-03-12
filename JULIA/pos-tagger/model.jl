using Knet, JLD
using AutoGrad: cat1d
include("preprocess.jl")
# Let's work with English-Lines Data
trainfile = "/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-train.conllu"
devfile   = "/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-dev.conllu"
lm_model = "/ai/data/nlp/conll17/competition/chmodel_converted/english_chmodel.jld"

# Initializations
# optimization parameter creator for parameters
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)



# using AutoGrad
# let cat_r = recorder(cat); global vcatn, hcatn
#     function vcatn(a...)
#         if any(x->isa(x,Rec), a)
#             cat_r(1,a...)
#         else
#             vcat(a...)
#         end
#     end
#     function hcatn(a...)
#         if any(x->isa(x,Rec), a)
#             cat_r(2,a...)
#         else
#             hcat(a...)
#         end
#     end
# end


# xavier initialization
function initx(d...; ftype=Float32)
    if gpu() >=0
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end
end


function initr(d...; ftype=Float32, GPUFEATS=false)
    if gpu() >=0 && GPUFEATS
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end    
end


function initmodel(featdim, hiddens, dpostag)
    posnum = 17 # number of unique postags
    model = Any[]
    # initialize postags
    for (k, n, d) in ((:postag, posnum, dpostag),) 
        push!(model, [ initr(d) for i in 1:n ])
    end

    # initialize MLP
    mlpdims = (featdim, hiddens..., posnum) # we will make postag preds
    tagger = Any[]
    for i in 2:length(mlpdims)
        push!(tagger, initx(mlpdims[i], mlpdims[i-1])) # w
        push!(tagger, initx(mlpdims[i], 1)) # b
    end
    push!(model, tagger)
end


# To calcute the features ±3 tokens centered at the current token (wptr)
function winwords(wptr::Number, slen::Int64, wlen::Int64)
    if wptr + wlen > slen
        toright = (wptr+1):slen
    else
        toright = (wptr+1):(wptr+wlen)
    end

    if wptr - wlen <= 0
        toleft = 1:(wptr-1)
        #toleft = wptr-1:-1:1
    else
        #toleft = wptr-1:-1:(wptr-wlen) reverted!
        toleft = (wptr-wlen):(wptr-1)
    end
    return toleft, toright
end


# Possible features for a word in the buffer
# 1. word, forw and back vectors (350 + 300 + 300) in window sized 3
# 2. At most 4 previous predicted pos-tags (posdim * 4)
function features(model, taggers, feats, wlen=3)
    pvec0 = zeros(model[1])
    wvec0 = zeros(taggers[1].sent.wvec[1])
    fvec0 = bvec0 = zeros(taggers[1].sent.fvec[1])
    fmatrix = []
    for t in taggers
        w = (isdone(t) ? length(t.sent) : t.wptr) # This will fix done issue!

        (tL, tR) = winwords(w, length(t.sent), wlen)
        if 'v' in feats
            lcount = length(tL)
            for i in 1:(wlen-lcount)
                push!(fmatrix, wvec0)
            end
            for i in tL
                push!(fmatrix,t.sent.wvec[i])
            end

            push!(fmatrix, t.sent.wvec[w])

            rcount=length(tR)
            for i in tR
                push!(fmatrix, t.sent.wvec[i])
            end
            for i in 1:(wlen-rcount)
                push!(fmatrix, wvec0)
            end
        end

        if 'c' in feats
            lcount = length(tL)
            for i in 1:(wlen-lcount)
                push!(fmatrix, fvec0)
                push!(fmatrix, bvec0)
            end
            for i in tL
                push!(fmatrix, t.sent.fvec[i])
                push!(fmatrix, t.sent.bvec[i])
            end
            push!(fmatrix, t.sent.fvec[w])
            push!(fmatrix, t.sent.bvec[w])

            rcount=length(tR)
            for i in tR
                push!(fmatrix, t.sent.fvec[i])
                push!(fmatrix, t.sent.bvec[i])
            end
            for i in 1:(wlen-rcount)
                push!(fmatrix, fvec0)
                push!(fmatrix, bvec0)
            end
        end

        # prev-postag
        p = length(t.preds)
        prevs = (p >= 4 ? ((p-4+1):p) : 1:p)
        for i in prevs
            push!(fmatrix, model[t.preds[i]])
        end
        s = 4-length(prevs)
        for i in 1:s
            push!(fmatrix, pvec0)
        end
    end
    fmatrix = cat1d(fmatrix...)
    ncols = length(taggers)
    nrows = div(length(fmatrix), ncols)
    return reshape(fmatrix, nrows, ncols)
end

# All uggly 1s is not to overwrite Knet's predefined functions!!

function mlp1(w, input;pdrop=(0.0, 0.0)) 
    x = dropout(input, pdrop[1])
    for i in 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
        x = dropout(x, pdrop[2])
    end
    return w[end-1]*x .+ w[end]
end




function oracleloss(model, sentences, feats; lval=[], pdrop=(0.0, 0.0))
    taggers = map(Tagger, sentences)
    taggersdone = map(isdone, taggers)
    totalloss = 0.0
    featmodel, mlpmodel = model[1], model[2]

    while !all(taggersdone)
        fmatrix = features(featmodel, taggers, feats)
        if gpu() >= 0
            fmatrix = KnetArray(fmatrix)
        end
        scores = mlp1(mlpmodel, fmatrix, pdrop=pdrop)
        logprobs = logp(scores, 1)
        for (i, t) in enumerate(taggers)
            taggersdone[i] && continue
            goldind = goldtag(t)
            cval = logprobs[goldind, i]
            totalloss -= cval
            move!(t, goldind)
            (isdone(t) ? taggersdone[i]=true : nothing)
        end
    end
    ntot = mapreduce(length, +, 0, sentences)
    push!(lval, (AutoGrad.getval(totalloss)/ntot))
    return totalloss / ntot
end

oraclegrad = grad(oracleloss)


function oracletrain(model, corpus, feats, opts, batchsize, lval=[]; pdrop=nothing)
    sentbatches = minibatch1(corpus, batchsize)
    lval = []
    for sentences in sentbatches
        ograds = oraclegrad(model, sentences, feats, lval=lval, pdrop=pdrop)
        update!(model, ograds, opts)
    end
    avgloss = mean(lval)
    return avgloss
end


function oracleacc(model, corpus, feats, batchsize; pdrop=(0.0, 0.0))
    sentbatches = minibatch1(corpus, batchsize)
    ntot = ncorr = 0
    for sentences in sentbatches
        taggers = map(Tagger, sentences)
        taggersdone = map(isdone, taggers)
        totalloss = 0.0
        featmodel, mlpmodel = model[1], model[2]

        while !all(taggersdone)
            fmatrix = features(featmodel, taggers, feats)
            if gpu() >= 0
                fmatrix = KnetArray{Float32}(fmatrix)
            end
            scores = Array(mlp1(mlpmodel, fmatrix, pdrop=pdrop))
            (_, indx) = findmax(scores, 1)
            #logprobs = logp(scores, 1) # no need to logprobability
            for (i, t) in enumerate(taggers)
                taggersdone[i] && continue
                pred_tag = indx[i] - (i-1) * 17 # To get cartesian index

                # goldind = goldtag(t);#move!(t, goldind)
                #cval = logprobs[goldind, i]
                #totalloss -= cval
                move!(t, UInt8(pred_tag))
                if isdone(t)
                    taggersdone[i] = true
                    ncorr += sum(t.preds .== t.sent.postag)
                end
                #(isdone(t) ? taggersdone[i]=true : nothing)
            end
        end
        ntot += mapreduce(length, +, 0, sentences)
    end
    return ncorr / ntot
end


function main()
    # Load data
    #corpus = load_conllu(trainfile)

    # dbg 
    corpus = load_conllu(trainfile)
    dev = load_conllu(devfile)

    # fill context and word embeddings from  pre-trained model
    bundle = load_lm(lm_model)
    fillallvecs!(corpus, bundle); fillallvecs!(dev, bundle);

    # Initialize model
    feats = "cv"
    batchsize = 2
    POSEMBEDDINGS = 128
    featdim = 7162 # 300*14 + 350*7 + 128*4 (context + word + embedding vectors)
    hiddens = [2048]
    model = initmodel(featdim, hiddens, POSEMBEDDINGS)
    opts = oparams(model, Adam; gclip=5.0)

    acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
    println("Initial dev accuracy $acc1")
    for i in 1:10
        lval = []
        lss = oracletrain(model, corpus, feats, opts, batchsize, lval; pdrop=(0.5, 0.5))
        trnacc = oracleacc(model, corpus, feats, batchsize; pdrop=(0.0, 0.0))
        acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
        println("Loss val $lss trn acc $trnacc tst acc $acc1 ...")
    end


    
end


