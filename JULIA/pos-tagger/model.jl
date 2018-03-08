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



function initmodel(featdim, hiddens)
    posnum = 17 # number of postags
    model = Any[]
    # initialize postags
    for (k, n, d) in ((:postag, posnum, dpostag)) 
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

function main()
    # Load data
    corpus = load_conllu(trainfile)




end
