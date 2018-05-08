include("train.jl")

function input(prompt::String="")
    print(prompt)
    return chomp(readline())
end

function predme(seqid, imds; resmodel=nothing, bundle=nothing)

    # for debug purposes
    #seqid = 31011385
    #imds = [5,2,4,1]
    sort!(imds)

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
    return indx
end


function naive_suggest(catvocab, indx, top, suggest_text)
    result = nothing
    for i in 1:top
        my_items = get(catvocab, indx[i], 46) # Todo: fix hardcoded
        c1 = find(x->contains(x, suggest_text), my_items)
        if !isempty(c1)
            return my_items[first(c1)]
        end
    end
    return result
end


function main_pred(;bundle=nothing, top=5)
    metadata = JSON.parsefile("data/label/train_no_dup.json")
    if bundle == nothing
        bundle = JLD.load("600_all_best_model.jld")
    end
    resmodel = create_resmodel()
    seqid = input("What is image id: ")
    find(x->x["set_id"]==parse(seqid))
    imds  = input("Enter the image ")
    imds = map(parse, split(imds))
    catvocab = create_catvocab("data/label/train_no_dup.json")
    indx = predme(seqid, imds, resmodel=resmodel, bundle=bundle)
    s1 = input("Do you want to enter text: ")
    if "y" || "yes" == s1
        sequence_text = input("What do you want: ")
    end
    return indx, catvocab, metadata
end


