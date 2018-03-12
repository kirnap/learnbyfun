#include("main_v6.jl")
#chm="/ai/data/nlp/conll17/competition/chmodel_converted/english_chmodel.jld"
#d1="/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-train.conllu"
#d2="/ai/data/nlp/conll17/ud-treebanks-v2.0/UD_English-LinES/en_lines-ud-dev.conllu"

function omerbeamtrain(optim, model, feats, corpus, arctype, beamwidth)
    sentbatches = minibatch(corpus, 1; maxlen=MAXSENT, minlen=MINSENT, shuf= true) # batchsize 1 for now
    if LOGGING > 0
        nsent = sum(map(length,sentbatches)); nsent0 = length(corpus)
        nword = sum(map(length,vcat(sentbatches...))); nword0 = sum(map(length,corpus))
        @msg("nsent=$nsent/$nsent0 nword=$nword/$nword0")
    end
    nwords = StopWatch()
    nsteps = Any[0,0]

    for sentences in sentbatches
        grads = beamgrad2(model, feats, sentences, arctype, beamwidth)
        update!(model, grads, optim)
        nw = sum(map(length,sentences))
        speed = inc(nwords, nw)
        # if speed != nothing
        #     date("$(nwords.ncurr) words $(round(Int,speed)) wps $(round(Int,100*nsteps[1]/nsteps[2]))% steps")
        #     nsteps[:] = 0
        # end
    end
    println()
end

# Beam representation2 : (scoreval,  moves, isstop, isgold, parser)
function beamloss2(model, feats, sentences, arctype, beamwidth)
    count = 0
    parser = first(map(arctype, sentences)) # batchsize 1
    cbeams = Any[(0.0, Int[], false, false, parser)] # initialize beam
    gpath = get_gpath(parser)
    totloss = 0.0
    # get previous beam if gold falled out
    prevbeams = cbeams
    while !allstop(cbeams) # when all branches stopped get the loss
        cbeams = move_beam(model, feats, prevbeams, gpath, beamwidth)
        if containsgold(cbeams)
            prevbeams = cbeams
        else # gold fallen
            break
        end
    end

    # That is temprorary solution to being falled from beam
    goldin = find(x->branchgold(x), prevbeams)
    if length(prevbeams) == 1 # That means goldpath fallen initially
        count += 1; (count > 20 ? Base.warn_once("beam fallen for $count") : nothing);
        goldin = [1] # we may add gold path manually?
        prevbeams = cbeams
    end


    featmodel, mlpmodel = splitmodel(model)

    all_parsers = [ copy(parser) for i in 1:length(prevbeams) ]
    all_acts = [ i[2] for i in prevbeams ]
    totscores = Any[]
    for i in 1:length(all_acts[1])
        indx = map(k->all_acts[k][i]+(k-1)*73, 1:length(all_acts)) # 73 is ugly but ok for now
        fmatrix = features(all_parsers, feats, featmodel)
        if gpu()>=0; fmatrix = KnetArray(fmatrix); end
        mscores = mlp(mlpmodel, fmatrix)
        push!(totscores, reshape(mscores[indx], length(prevbeams), 1))
        for (p, m) in zip(all_parsers, map(x->x[i], all_acts)); move!(p, m);end;
    end
    t1 = sum(hcat(totscores...), 2)
    totloss += nll(t1, goldin)
    return totloss / length(all_acts[1])
end

beamgrad2 = grad(beamloss2)

parserstate(member)=member[end]
branchstop(member)=member[end-2]
branchgold(member)=member[end-1]
containsgold(p::Array{Any, 1})=any(map(x->x[end-1],p)) 
ingold(v::Array{Int}, big::Array{Int})=(v == view(big, 1:length(v)))
allstop(allchildren::Array{Any, 1})=all(map(branchstop, allchildren))

function move_beam(model, feats, cbeams, gpath, beamwidth)
    ret = Any[]
    featmodel, mlpmodel = splitmodel(model)
    for beam in cbeams
        if branchstop(beam); push!(ret, beam); continue; end # That beam stopped do not touch staying the same

        sc1, acts, isstop, isgold, p1 = beam # open up beam
        
        if !anyvalidmoves(p1) # no valid moves
            push!(ret, (sc1, acts, true, isgold, p1))
            continue
        end

        fmatrix = features([p1], feats, featmodel)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix): fmatrix)
        mscores = mlp(mlpmodel, fmatrix)
        sindx = 73 # ugly but we now the number of moves

        # get available moves
        mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
        v1 = mcosts .!= typemax(Cost)
        vals = collect(enumerate(v1)) # move states for a parser (movenum, bool)

        for s in 1:sindx
            if (s, true) in vals # valid move
                p_new = copy(p1); move!(p_new, s);new_act = copy(acts);push!(new_act, s);
                cand = (sc1 + getval(mscores)[s], new_act, isstop, ingold(new_act, gpath), p_new)
                push!(ret, cand)
            end
        end
    end
    n = (length(ret) >= beamwidth ? beamwidth : length(ret)) # there may be less valid children
    return sort!(ret, by=x->x[1], rev=true)[1:n] # sort them and get n bests
end


function get_gpath(pgiven)
    parser = copy(pgiven)
    moveset = Int[]
    mcosts = Array{Cost}(parser.nmove) 
    movecosts(parser, parser.sentence.head, parser.sentence.deprel, mcosts) # movecosts
    gmove = indmin(mcosts) 
    while mcosts[gmove] != typemax(Cost)
        push!(moveset, gmove)
        move!(parser, gmove)
        movecosts(parser, parser.sentence.head, parser.sentence.deprel, mcosts) # movecosts
        gmove = indmin(mcosts) # get gold move
    end
    return moveset
end





#= DEAD Code
# Beam representation :  (scoreval, score, moves, isstop, isgold, parser)
### That implementation for test and speed up purposes
function move_beam2(model, feats, cbeams, gpath, beamwidth)
    ret = Any[]
    featmodel, mlpmodel = splitmodel(model)
    for beam in cbeams
        if beamstop(beam); push!(ret, beam); continue; end # That beam stopped do not touch staying the same

        sc1, scg, acts, isstop, isgold, p1 = beam # open up beam
        
        # if !anyvalidmoves(p1) # no valid moves
        #     push!(ret, (sc1, scg, acts, true, isgold, p1))
        #     continue
        # end

        fmatrix = features([p1], feats, featmodel)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix): fmatrix)
        mscores = Array(mlp(mlpmodel, fmatrix))

        # no need for sort in the next version
        sindx = sortperm(reshape(getval(mscores), length(mscores)), rev=true) # sorted moves via score

        # no need for that kind of check anyvalidmoves(p) is ok in the next version
        # get available moves
        mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
        v1 = mcosts .!= typemax(Cost)
        vals = collect(enumerate(v1)) # move states for a parser (movenum, bool)

        n = sum(v1)
        if n == 0 # no valid moves
            push!(ret, (sc1, scg, acts, true, isgold, p1))
            continue
        end

        for s in sindx
            if (s, true) in vals # valid move
                p_new = copy(p1); move!(p_new, s);new_act = copy(acts);push!(new_act, s);
                cand = (sc1 + getval(mscores)[s], scg + mscores[s], new_act, isstop, ingold(new_act, gpath), p_new)
                push!(ret, cand)
            end
        end
    end
    n = (length(ret) >= beamwidth ? beamwidth : length(ret)) # there may be less valid children
    return sort!(ret, by=x->x[1], rev=true)[1:n] # sort them and get bests
end


### That implementation for test and speed up purposes
function move_beam3(model, feats, cbeams, gpath, beamwidth)
    ret = Any[]
    featmodel, mlpmodel = splitmodel(model)
    for beam in cbeams
        if beamstop(beam); push!(ret, beam); continue; end # That beam stopped do not touch staying the same

        sc1, scg, acts, isstop, isgold, p1 = beam # open up beam
        
        # if !anyvalidmoves(p1) # no valid moves
        #     push!(ret, (sc1, scg, acts, true, isgold, p1))
        #     continue
        # end

        fmatrix = features([p1], feats, featmodel)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix): fmatrix)
        mscores = Array(mlp(mlpmodel, fmatrix))
        sindx = sortperm(reshape(getval(mscores), length(mscores)), rev=true) # sorted moves via score

        # get available moves
        mcosts = movecosts(p1, p1.sentence.head, p1.sentence.deprel)
        v1 = mcosts .!= typemax(Cost)
        vals = collect(enumerate(v1)) # move states for a parser (movenum, bool)

        n = sum(v1)
        if n == 0 # no valid moves
            push!(ret, (sc1, scg, acts, true, isgold, p1))
            continue
        end

        for s in sindx
            if (s, true) in vals # valid move
                p_new = copy(p1); move!(p_new, s);new_act = copy(acts);push!(new_act, s);
                cand = (sc1 + getval(mscores)[s], scg + mscores[s], new_act, isstop, ingold(new_act, gpath), p_new)
                push!(ret, cand)
            end
        end
    end
    n = (length(ret) >= beamwidth ? beamwidth : length(ret)) # there may be less valid children
    return sort!(ret, by=x->x[1], rev=true)[1:n] # sort them and get bests
end



function beam_try(model, sentences, arctype, feats, beamwidth) # assume batchsize 1
    
    parsers = map(arctype, sentences) # you have parsers of batchsize;
    
    fmatrix = features(parsers, feats, featmodel) # input to mlp;
    fmatrix = (gpu() >= 0 ? KnetArray(fmatrix) : Array(fmatrix))
    mscores = Array(mlp(mlpmodel, fmatrix))
    golds = gold_path(parsers[1]) # get gold path

    newbeam = create_beam(mscores, parsers[1], beamwidth) # Initial beam
    #tostop = (golds[1] in bmoves) #  what if the gold move falls out from the beam in the beginning ?
end



function create_beam(model, feats, beamp)#(mscores, parser, beamwidth)
    sindx = sortperm(reshape(getval(mscores), length(mscores)), rev=true) # sorted moves via score
    n = (sum(v1) >= beamwidth ? beamwidth : sum(v1)) # There may be less than beamwidth valid moves
    if n == 0
        return []
    end
    bmoves = Int[]
    for s in sindx
        if (s, true) in vals # validate
            push!(bmoves, s)
        end
        (length(bmoves) == n) && break
    end

    newbeam = [ (getval(mscores)[i], mscores[i], copy(parser)) for i in 1:n ]
    map(i->move!(newbeam[i][end], bmoves[i]), 1:n)
    return newbeam
end


function next_beam(featmodel, mlpmodel, cbeam, feats, beamwidth; earlystop=true)
    beamset = Any[]
    for beam in cbeam
        fmatrix = features([beam[end]], feats, featmodel)
        fmatrix = (gpu() >= 0 ? KnetArray(fmatrix) : Array(fmatrix))
        mscores = Array(mlp(mlpmodel, fmatrix))
        bcnd = create_beam(mscores, beam[end], beamwidth)
        push!(beamset, bcnd)
    end
    return beamset
end


function get_gpath(pgiven)
    parser = copy(pgiven)
    moveset = Int[]
    mcosts = Array{Cost}(parser.nmove) 
    movecosts(parser, parser.sentence.head, parser.sentence.deprel, mcosts) # movecosts
    gmove = indmin(mcosts) 
    while mcosts[gmove] != typemax(Cost)
        push!(moveset, gmove)
        move!(parser, gmove)
        movecosts(parser, parser.sentence.head, parser.sentence.deprel, mcosts) # movecosts
        gmove = indmin(mcosts) # get gold move
    end
    return moveset
end
=#

# I am gonna play with that line
#train("--load $chm --datafiles $d1 $d2 --otrain 1 --btrain 0  --seed 4")
