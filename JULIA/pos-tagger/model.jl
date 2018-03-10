using Knet, JLD
using AutoGrad: cat1d
include("preprocess.jl")
# Let's work with English-Lines Data
trainfile = "/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-train.conllu"
devfile   = "/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-dev.conllu"
lm_model = "/ai/data/nlp/conll17/competition/chmodel_converted/english_chmodel.jld"

# Initializations

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
function winwords(wptr::Int32, slen::Int64, wlen::Int64)
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
        w = t.wptr
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


function main()
    # Load data
    #corpus = load_conllu(trainfile)

    # dbg 
    corpus = load_conllu("foo2.conllu")

    # fill context and word embeddings from  pre-trained model
    bundle = load_lm(lm_model)
    fillallvecs!(corpus, bundle)

    # Initialize model
    POSEMBEDDINGS = 128
    featdim = 950 + POSEMBEDDINGS*4 # We have much more features
    hiddens = [2048]
    all_model = initmodel(featdim, hiddens, POSEMBEDDINGS)
    return (all_model,corpus)
end


