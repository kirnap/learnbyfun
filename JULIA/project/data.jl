# ~9k word has more than 1 phonetic value

# define global constants
const SOW = Char(0x0012) # word start
const EOW = Char(0x0013) # word ends
const PAD = Char(0x0014) # encoder pad
const SOP = "<sop>"  # phoneme start
const EOP = "</sop>" # phone end

# To visualize what is happening
# const PAD = '⋮';const SOW = '↥';const EOW = 'Ϟ'


# Implement all preprocessing related
struct s2sData
    encoding
    decoding
end

# Read all the data w/o minibatching
function createData(filename::AbstractString)
    toSkip = ["(2)", "(3)", "(4)"]# Ugly preprocessing hack but works, mapreduce line uses this array

    envocab = Dict{Char, UInt16}(SOW=>0x0001, EOW=>0x0002, PAD=>0x0003)
    devocab = Dict{String, UInt16}(EOP=>0x0001, SOP=>0x0002) # SOP=>0x0002 for start phoneme
    data = []
    for line in eachline(filename)
        items = split(line)
        w = items[1]
        (mapreduce(t->contains(w, t), |, false, toSkip) && continue)
        chars = UInt16[]
        push!(chars, envocab[SOW])
        for c in w
            s = get!(envocab, c, 1+length(envocab))
            push!(chars, s)
        end
        push!(chars, envocab[EOW])
        
        phons = UInt16[]
        push!(phons, devocab[SOP]) #use sop
        for p in items[2:end]
            s = get!(devocab, p, 1+length(devocab))
            push!(phons, s)
        end
        push!(phons, devocab[EOP])
        
        push!(data, s2sData(chars, phons))
    end
    return data, envocab, devocab
end


function mb(data, batchsize)
    sents = sort(data, by=x->length(x.encoding), rev=true)
    ret = Any[]; 
    for i in 1:batchsize:length(sents)
        j = min(i+batchsize-1, length(sents))
        sij = view(sents, i:j)
        T = length(sij[1].encoding)
        mx = UInt16[]; bs = UInt16[]; my=Array{UInt16}[];
        for t in 1:T
            b = 0
            for s in sij
                if t == 1
                    push!(my, s.decoding)
                    push!(mx, s.encoding[t])
                    b += 1
                elseif t<=length(s.encoding)
                    push!(mx, s.encoding[t])
                    b += 1
                else
                    break
                end
            end
            push!(bs, b)
        end

        # Decoder-side
        Td = maximum(length, my)
        mys = UInt16[]; bsd = UInt16[];
        for t in 1:Td
            b = 0
            for d in my
                if t <= length(d)
                    push!(mys, d[t])
                    b += 1
                else
                    continue
                end
            end
            push!(bsd, b)
        end
        enc = (mx, bs); dec0 = (mys, bsd);
        push!(ret, (enc, dec0))
    end
    return ret
end


# encoder sequence padded, decoder sequence in reversed order 
function mb2(data, batchsize)
    sents = sort(data, by=x->length(x.encoding), rev=true)
    ret = Any[]
    for i in 1:batchsize:length(sents)
        j = min(i+batchsize-1, length(sents))
        (j+1-i != batchsize) && break # not to use surplus batch
        sij = sort(view(sents, i:j), by=x->length(x.decoding), rev=true)
        T = length(sij[1].encoding)
        mx = UInt16[]; bs = UInt16[]; my=Array{UInt16}[];

        # encoder-side
        for t in 1:T
            b = 0
            for s in sij
                if t == 1
                    push!(my, s.decoding)
                    push!(mx, s.encoding[t])
                    b += 1
                elseif t<=length(s.encoding)
                    push!(mx, s.encoding[t])
                    b += 1
                else
                    push!(mx, 0x0003) # pad character
                    b+=1
                end
            end
            push!(bs, b)
        end
        # decoder-side
        Td = maximum(length, my)
        mys = UInt16[]; bsd = UInt16[]; mygolds = UInt16[];
        for t in 1:Td
            b = 0
            for d in my
                if t <= length(d)
                    push!(mys, d[t])
                    b += 1
                else
                    continue
                end
            end
            push!(bsd, b)
        end
        enc = (mx, bs); dec0 = (mys, bsd, mygolds);
        push!(ret, (enc, dec0))
    end
    return ret
end
