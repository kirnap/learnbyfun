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
       #x = relu.(w[i] * x .+ w[i+1]) # To test in linear models
        x = w[i] * x .+ w[i+1]
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

# Some beam search utility functions
branchstop(member) = member[3] 
branchgold(member) = member[end-1]
taggerof(member)   = member[end]
ingold(v::Array{UInt8}, big::Array{UInt8})=(v==view(big, 1:length(v)))
containsgold(p::Array{Any, 1})=any(map(branchgold, p))
allstop(p::Array{Any, 1})=all(map(branchstop, p))
goldstate(p::Array{Any, 1})=(for b in p; if branchgold(b); return b;end;end;error("Not constains gold branch but called goldstate!"))

# Calculate the number of steps that are correctly done
function goldcovered(pbeam::Array{Any,1}, gpath::Array{UInt8,1})
    goldin = find(x->branchgold(x), pbeam)
    @assert length(goldin) == 1 # No more than single beam
    goldin = goldin[1]
    goldmember = pbeam[goldin]
    gtpreds = goldmember[end].preds # array of gold moves

    if length(gtpreds) == length(gpath) # all path covered
        return true
    end
    return false # you need to give next step manually
end

# Beam representation: (scoreval, moves, isstop, isgold, tagger)
function beamloss(model, feats, sentences, beamwidth; lval=[], pdrop=nothing)
    totloss = 0.0
    tagger = Tagger(first(sentences)) # Batchsize is 1
    cbeams = Any[(0.0, UInt8[], false, false, tagger)]
    gpath = goldpath(tagger)

    prevbeams = cbeams
    while !allstop(cbeams)
        cbeams = movebeam(model, feats, prevbeams, gpath, beamwidth, pdrop=pdrop)
        if containsgold(cbeams)
            prevbeams = cbeams
        else # gold fallen
            break
        end
    end

    # we need to put next step manually
    

    featmodel, mlpmodel = model[1], model[2]
    
    # 1. You need to normalize over paths based on n best value
    # 2. If gold path fallen out initially, add manually
    if length(prevbeams) == 1 # goldpath fallen initially?
        print("+"); flush(STDOUT);# To warn myself

        #omerdbg[:prevbeams2] = prevbeams; ######### DBG
        desired = (0.0, UInt8[tagger.sent.postag[1]], true, true, copy(tagger))
        push!(cbeams, desired)
        prevbeams = cbeams
    # we need to take a look state of gold branch
    elseif !goldcovered(prevbeams, gpath) # gold path fallen after some forward going
        gdesired = goldstate(prevbeams) # we need to move one-more step
        gscoreval, gacts, gisstop, gisgold, gtagger = gdesired
        goldmove = gtagger.sent.postag[length(gtagger.preds)+1] # an expected move to take in the next step

        
        # manually calculate the score values
        fmatrix = features(featmodel, [gtagger], feats)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix) : fmatrix)
        scores = mlp1(mlpmodel, fmatrix, pdrop=pdrop) # size(scores) 17,1
        
        t_new = copy(gtagger); move!(t_new, goldmove); new_act = copy(gacts); push!(new_act, goldmove);
        desired = (gscoreval+scores[Int(goldmove)], new_act, isdone(t_new), true, t_new)
        push!(cbeams, desired) # manually put the correct one

        # global omer = (prevbeams, cbeams, model, feats, prevbeams, gpath, beamwidth, pdrop, sanity); error("uzucu errors are not finishing anymore"); # dbg
        prevbeams = cbeams
    else
        # everything is ok keep going
    end


    goldin = find(x->branchgold(x), prevbeams)
    tgold = prevbeams[goldin[1]][end]
    numofpred = length(tgold.preds)
    if length(prevbeams[goldin[1]][end].preds) != length(tagger.sent.postag) # all path is not covered
        print(".");flush(STDOUT);
    end

    if length(goldin) != 1
        #omerdbg[:prevbeams] = prevbeams; omerdbg[:model] = model; omerdbg[:cbeams] = cbeams;
        #omerdbg[:gpath] = gpath; omerdbg[:beamwidth] = beamwidth; omerdbg[:feats] = feats; omerdbg[:pdrop]=pdrop;
        error("uzucu bir error")
    end
    goldin = goldin[1]
    #@assert length(goldin) == 1; goldin = goldin[1] # sanity check
    all_paths = [ i[2] for i in prevbeams ]
    all_taggers = [ copy(tagger) for i in 1:length(prevbeams) ]
    ntot = length(prevbeams)
    totscores = Any[]
    for i in 1:length(all_paths[goldin]) # only move through gold path
        indx = map(k->all_paths[k][i]+(k-1)*17, 1:ntot) # convert cartesian index ugly 17!
        fmatrix = features(featmodel, all_taggers, feats)
        if gpu() >= 0;fmatrix = KnetArray(fmatrix);end;
        scores = mlp1(mlpmodel, fmatrix, pdrop=pdrop)
        push!(totscores, reshape(scores[indx], length(prevbeams), 1))
        for (ti, mi) in zip(all_taggers, map(x->x[i], all_paths)); move!(ti, mi);end;
    end
    # normalize over different paths
    t1 = sum(hcat(totscores...), 2)
    totloss -= logp(t1, 1)[goldin]
    push!(lval, getval(totloss)/length(all_paths[goldin]))
    return totloss / length(all_paths[goldin]) # normalize with path length
end

beamgrad = grad(beamloss)

function movebeam(model, feats, cbeams, gpath, beamwidth; pdrop=nothing)
    ret = Any[]
    featmodel, mlpmodel = model[1], model[2]

    for beam in cbeams
        if branchstop(beam); push!(ret, beam); continue;end; # Don't touch stopped beams

        scoreval, acts, isstop, isgold, tagger = beam

        if isdone(tagger)
            push!(ret, (scoreval, acts, true, isgold, tagger))
            continue
        end

        fmatrix = features(featmodel, [tagger], feats)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix) : fmatrix)
        scores = mlp1(mlpmodel, fmatrix, pdrop=pdrop) # size(scores) 17,1


        # Every move is available, no need for cartesian change: batchsize 1
        for i in 1:17 # 17 UPOSTAG
            t_new = copy(tagger); move!(t_new, UInt8(i)); new_act = copy(acts); push!(new_act, UInt8(i));
            cand = (scoreval + scores[i], new_act, isstop, ingold(new_act, gpath), t_new)
            push!(ret, cand)
        end
    end
    n = (length(ret) >= beamwidth ? beamwidth : length(ret))
    return sort!(ret, by=x->x[1], rev=true)[1:n] # get n best 
end


function oraclemain(;epochs=40)
    # Load data
    #corpus = dev =load_conllu("foo4.conllu") ;savemode=true;#dbg
    corpus = load_conllu(trainfile); dev = load_conllu(devfile); savemode=false
    
    # fill context and word embeddings from  pre-trained model
    bundle = load_lm(lm_model)
    fillallvecs!(corpus, bundle); fillallvecs!(dev, bundle);

    # Initialize model
    feats = "cv"
    batchsize = 16
    POSEMBEDDINGS = 128#256
    featdim =  300*14 + 350*7 + POSEMBEDDINGS*4 #(context + word + embedding vectors)
    hiddens = [2048]
    model = initmodel(featdim, hiddens, POSEMBEDDINGS)
    opts = oparams(model, Adam; gclip=5.0)

    acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
    println("Initial dev accuracy $acc1")
    for i in 1:epochs
        lval = []
        lss = oracletrain(model, corpus, feats, opts, batchsize, lval; pdrop=(0.5, 0.8))
        trnacc = oracleacc(model, corpus, feats, batchsize; pdrop=(0.0, 0.0))
        acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
        if savemode
            JLD.save("pos_experiment.jld", "model", model, "optims", opts)
            println("Loss val $lss trn acc $trnacc tst acc $acc1 ...")
            i==5 && break
        end
        !savemode && println("Loss val $lss trn acc $trnacc tst acc $acc1 ...")
    end
end


function beamtrain(model, corpus, feats, opts, beamwidth; pdrop=nothing)
    lval = []
    for sentences in corpus # batchsize 1
        bgrads = beamgrad(model, feats, [sentences], beamwidth; lval=lval, pdrop=pdrop)
        update!(model, bgrads, opts)
    end
    avgloss = mean(lval)
    return avgloss
end


function beammain(;prevmodel=nothing)
    # load-data
    info("reading data")
    corpus, dev = load_conllu(trainfile), load_conllu(devfile)
    fillallvecs!(corpus, load_lm(lm_model)); fillallvecs!(dev, load_lm(lm_model));
    
    # dbg
    #corpus = load_conllu("foo4.conllu"); fillallvecs!(corpus, load_lm(lm_model));prevmodel="pos_experiment.jld"
    # Initmodel
    beamwidth = 16; feats="cv";batchsize=16; println("Beamwidth $beamwidth"); flush(STDOUT);
    if prevmodel == nothing
        info("Initializing model...")
        POSEMBEDDINGS = 128
        featdim = 300*14 + 350*7 + POSEMBEDDINGS*4 #(context + word + embedding vectors)
        hiddens = [2048]
        model = initmodel(featdim, hiddens, POSEMBEDDINGS)
        opts = oparams(model, Adam; gclip=5.0)
    else
        model, opts = JLD.load(prevmodel, "model", "optims")
    end
    acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
    println("Initial dev accuracy $acc1")
    info("Oracle training..."); flush(STDOUT);
    lss = oracletrain(model, corpus, feats, opts, batchsize, []; pdrop=(0.5, 0.8))
    trnacc = oracleacc(model, corpus, feats, batchsize; pdrop=(0.0, 0.0))
    acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
    println("Loss val $lss trnacc $trnacc tst acc $acc1")

    info("Beam training...");flush(STDOUT);pdrp=(0.0, 0.0);
    for i in 1:40
        lss = beamtrain(model, corpus, feats, opts, beamwidth, pdrop=pdrp)
        trnacc = oracleacc(model, corpus, feats, batchsize; pdrop=(0.0, 0.0))
        acc1 = oracleacc(model, dev, feats, batchsize; pdrop=(0.0, 0.0))
        println("\nLoss val $lss trn acc $trnacc tst acc $acc1 ...")
        flush(STDOUT);
    end
end

!isinteractive() && beammain()
