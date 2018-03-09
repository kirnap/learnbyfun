# To read the data from .conllu formatted file
const PosTag = UInt8 # 17 distinct universal POS-tag

struct Sent
    word::Array
    postag::Array
    wvec::Array
    fvec::Array
    bvec::Array
end

import Base:start, next, done, length

start(s::Sent)=(sptr=1; return sptr)
next(s::Sent, state)=(return (s.word[state], state+1))
done(s::Sent, state)=(state>length(s.word))
length(s::Sent)=length(s.word)

function Sent()
    Sent([],[],[],[],[])
end


mutable struct Tagger
    preds::Array
    sent::Sent
    wptr::Int
end


function Tagger(s::Sent)
    return Tagger([], s, 1)
end

import Base:copy, copy!
function copy!(dst::Tagger, src::Tagger)
    #dpreds = copy(src.preds)
    copy!(dst.preds, src.preds)
    return dst
end


copy(src::Tagger)=copy!(Tagger(similar(src.preds), src.sent, src.wptr), src)
moveok(t::Tagger)=(length(t.sent) >= t.wptr)
_move!(t::Tagger, m::PosTag) = (push!(t.preds, m);t.wptr+=1;return nothing)
move!(t::Tagger, m::PosTag)  = (moveok(t) ? _move!(t, m) : error("Not any valid moves"))


function Base.:(==)(t1::Tagger, t2::Tagger)
    for f in fieldnames(t1)
        (getfield(t1, f) != getfield(t2, f)) && (return false)
    end
    return true
end


function load_conllu(file::AbstractString; posvocab::Dict{String, PosTag}=UPOSTAG)
    corpus = Any[]; s = Sent();
    for line in eachline(file)
        if line == ""
            if length(s) <= 64 && length(s) >=2
                push!(corpus, s)
            end
            s = Sent()
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing
            #                id   word  lem  upos  xpos  feat head      deprel
            word = m.captures[1]; push!(s.word, word)
            p = m.captures[2]; postag = posvocab[p]
            push!(s.postag, postag)
        else
            # Not to say anything:)
        end
    end
    return corpus
end


function old_lstm(weight, bias, hidden, cell, input; mask=nothing)
    gates = weight * vcat(input, hidden) .+ bias
    H = size(hidden, 1)
    forget = sigm.(gates[1:H, :])
    ingate = sigm.(gates[1+H:2H, :])
    outgate = sigm.(gates[1+2H:3H, :])
    change = tanh.(gates[1+3H:4H, :])
    (mask != nothing) && (mask = reshape(mask, 1, length(mask)))

    cell = cell .* forget + ingate .* change
    hidden = outgate .* tanh.(cell)

    if mask != nothing
        hidden = hidden .* mask
        cell = cell .* mask
    end
    return (hidden, cell)
end


function charlstm(model, cembed, word, sowchar, eowchar, unc, chvocab)
    weight, bias = model[1], model[2]
    h0 = c0 = fill!(similar(bias, 350, 1),0)

    hidden, cell = old_lstm(weight, bias, h0, c0, cembed[:, sowchar])
    for w in word
        tok = get(chvocab, w, unc)
        (hidden, cell) = old_lstm(weight, bias, hidden, cell, cembed[:, tok])
    end
    (hidden, cell) = old_lstm(weight, bias, hidden, cell, cembed[:, eowchar])
    return hidden
end


function wordlstm(fmodel, bmodel, sentence, hsow, heow)
    fvecs = Array{Any}(length(sentence))

    # gpu-cpu conversion
    ftype = typeof(fmodel[1])
    
    # forward lstm
    wforw, bforw = fmodel[1], fmodel[2]
    h0 = c0 = fill!(similar(bforw, 300, 1),0)
    (hidden, cell) = old_lstm(wforw, bforw, h0, c0, hsow)
    fvecs[1] = hidden
    for i in 1:length(sentence)-1
        input = ftype(sentence.wvec[i])
        (hidden, cell) = old_lstm(wforw, bforw, hidden, cell, input)
        fvecs[i+1] = hidden
    end

    # backward lstm
    bvecs = Array{Any}(length(sentence))
    wback, bback = bmodel[1], bmodel[2]
    h0 = c0 = fill!(similar(bforw, 300, 1),0)
    (hidden, cell) = old_lstm(wback, bback, h0, c0, heow)
    bvecs[end] = Array(hidden)
    for i in length(sentence):-1:2
        input = ftype(sentence.wvec[i])
        (hidden, cell) = old_lstm(wback, bback, hidden, cell, input)
        bvecs[i-1] = hidden
    end
    return fvecs, bvecs
end


function _fillwvecs!(corpus, model, cembed, sowchar, eowchar, unc, chvocab, sow_word, eow_word; gpufeats=false)
    for sentence in corpus
        for word in sentence
            h = charlstm(model, cembed, word, sowchar, eowchar, unc, chvocab)
            h1 = (gpufeats ? h : Array(h))
            push!(sentence.wvec, h1)
        end
    end
    hsow = charlstm(model, cembed, sow_word, sowchar, eowchar, unc, chvocab)
    heow = charlstm(model, cembed, eow_word, sowchar, eowchar, unc, chvocab)
    return (hsow, heow)
end


function _fillcvecs!(corpus, fmodel, bmodel, hsow, heow; gpufeats=false)
    for sentence in corpus
        fvecs, bvecs = wordlstm(fmodel, bmodel, sentence, hsow, heow)
        f = (gpufeats ? KnetArray{Float32} : Array{Float32})
        map(i->push!(sentence.fvec, f(i)), fvecs)
        map(i->push!(sentence.bvec, f(i)), bvecs)
    end
end


function fillallvecs!(corpus, state)
    (cembed, cmodel, fmodel, bmodel, soc, eoc, unc, chvocab, sow, eow) = state
    hsow, heow = _fillwvecs!(corpus, cmodel, cembed, soc, eoc, unc, chvocab, sow, eow)
    _fillcvecs!(corpus, fmodel, bmodel, hsow, heow)
end


function load_lm(lmfile)
    _all = load(lmfile)
    chvocab = _all["char_vocab"]
    soc = chvocab[_all["sowchar"]]; eoc = chvocab[_all["eowchar"]]; 
    unc = chvocab[_all["unkchar"]]
    sow  = _all["sosword"]; eow=_all["eosword"];

    fmodel = (gpu() >= 0 ? map(x->convert(KnetArray, x), _all["forw"]) : _all["forw"])
    bmodel = (gpu() >= 0 ? map(x->convert(KnetArray, x), _all["back"]) : _all["back"])
    cmodel = (gpu() >= 0 ? map(x->convert(KnetArray, x), _all["char"]) : _all["char"])
    cembed = (gpu() >= 0 ? convert(KnetArray, _all["cembed"]) : _all["cembed"])

    state = (cembed, cmodel, fmodel, bmodel, soc, eoc, unc, chvocab, sow, eow)
    return state
end


# Universal POS tags (17)
const UPOSTAG = Dict{String,PosTag}(
"ADJ"   => 1, # adjective
"ADP"   => 2, # adposition
"ADV"   => 3, # adverb
"AUX"   => 4, # auxiliary
"CCONJ" => 5, # coordinating conjunction
"DET"   => 6, # determiner
"INTJ"  => 7, # interjection
"NOUN"  => 8, # noun
"NUM"   => 9, # numeral
"PART"  => 10, # particle
"PRON"  => 11, # pronoun
"PROPN" => 12, # proper noun
"PUNCT" => 13, # punctuation
"SCONJ" => 14, # subordinating conjunction
"SYM"   => 15, # symbol
"VERB"  => 16, # verb
"X"     => 17, # other
)
