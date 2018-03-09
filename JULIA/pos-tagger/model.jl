using Knet, JLD
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

function windim(wptr::Int, slen::Int, wlen::Int)
    if wptr + wlen > slen
        toright = (wptr+1):slen
    else
        toright = (wptr+1):(wptr+wlen)
    end
    return toright

    # TODO: implement toleft
    # extensive check for both toleft and toright
end

# Possible features for a word in the buffer
# 1. word, forw and back vectors (350 + 300 + 300) in window sized 3
# You need to calcute the features ±3 tokens centered at the current token (wptr)


# 2. At most 4 previous predicted pos-tags (posdim * 4)

function features(model, taggers, feats)
    pvec0 = zeros(model[1])
    fmatrix = []
    for t in taggers
        w = t.wptr
        if 'v' in feats
            push!(fmatrix, t.sent.wvec[w])
        end
        if 'c' in feats
            push!(fmatrix, t.sent.fvec[w])
            push!(fmatrix, t.sent.bvec[w])
        end
        
    end
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
    featdim = 950 + POSEMBEDDINGS*4
    hiddens = [2048]
    all_model = initmodel(featdim, hiddens, POSEMBEDDINGS)
    return all_model
end


