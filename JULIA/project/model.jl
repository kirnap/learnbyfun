wenc(model)    = model[1]
enc_r(model)   = model[2]
enc_wr(model)  = model[3]

wdec(model)    = model[4]
dec_r(model)   = model[5]
dec_wr(model)  = model[6]

enc_att(model) = model[7]
dec_att(model) = model[8]
alp_att(model) = model[9]

wout(model)    = model[10]
bout(model)    = model[11]

function initmodel(envsize, devsize, eembed, deembed, hidden, attn; rnnType=:lstm, dropout=(0.0, 0.0, 0.0))
    f = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
    w(d...) = f(xavier(Float32, d...))
    model = Any[]
    push!(model, w(eembed, envsize)) # encode embed

    # encoder lstm
    renc, wrenc = rnninit(eembed, hidden, rnnType=rnnType, numLayers=1)
    push!(model, renc) 
    push!(model, wrenc) 

    push!(model, w(deembed, devsize)) # decode embed

    # decoder lstm
    rdec, wrdec = rnninit(deembed, hidden, rnnType=rnnType, numLayers=1)
    push!(model, rdec) 
    push!(model, wrdec)

    # attention related weights
    push!(model, w(attn, hidden)) # key-vectors attention weights
    push!(model, w(attn, hidden)) # value vector attention weights
    push!(model, w(1, attn)) # attention MLP output side

    # output weight and bias
    push!(model, w(devsize, 2*hidden)) # output w
    push!(model, f(zeros(devsize, 1))) # output b
    return model
end

function soft(unalp)
    et1 = unalp .- maximum(unalp, 2)
    et3 = exp.(et1)
    return et3 ./ sum(et3, 2)
end


function encode(model, encx, encbs)
    wembed_e , renc, wrenc = wenc(model), enc_r(model), enc_wr(model)
    x_enc = wembed_e[:, encx]
    h_enc, hf, cf = rnnforw(renc, wrenc, x_enc, batchSizes=encbs, hy=true, cy=true)
    return (h_enc, hf, cf)
end


function predict(model, alldata)
    # encoder
    encdata, decdata = alldata
    encx, encbs = encdata
    h_enc, hf, cf = encode(model, encx, encbs)

    # attention weights
    w_att1, w_att2, w_atta = enc_att(model), dec_att(model), alp_att(model)
    H = size(h_enc)[1] # Hidden
    T = length(encbs); B = Int(encbs[1]); A = size(w_att1)[1] # Time, Batchsize, Attention

    # pre attention
    preatt1 = w_att1 * h_enc    
    preatt = reshape(preatt1, (A, B, T))
    

    # decoder
    decx, decbs = decdata
    wembed_d, rdec, wrdec = wdec(model), dec_r(model), dec_wr(model)

    # output
    wy, by = wout(model), bout(model)


    hf = reshape(hf, size(hf)[1], size(hf)[2])
    cf = reshape(cf, size(cf)[1], size(cf)[2])
    hprev = nothing
    cursor = 1; outs= Any[];
    for i in 1:length(decbs) # decoder time steps
        its = view(decx, cursor:(cursor+decbs[i]-1))
        hf = hf[:, 1:decbs[i]]
        cf = cf[:, 1:decbs[i]]

        hprev = hf

        # decode
        x_dec = wembed_d[:, its]
        h_dec, hf, cf = rnnforw(rdec, wrdec, x_dec, hf, cf, batchSizes=decbs[i:i], hy=true, cy=true)
        hf = reshape(hf, size(hf)[1], size(hf)[2])
        cf = reshape(cf, size(cf)[1], size(cf)[2])

        # attend
        query1 = w_att2 * hprev # hprev previous time step decoder
        query = hcat(query1, similar(query1, A, B-decbs[i]))
        q2 = reshape(query, (A, B, 1))
        q3 = tanh.(preatt2 .+ q2)
        q4 = reshape(q3, A, B*T)
        q5 = w_atta * q4
        q6 = reshape(q5, B, T)
        alphas = soft(q6)
        al1 = reshape(alphas, 1, B*T)
        context1 = h_enc .* al1
        context2 = reshape(context1, H, B, T)
        context3 = reshape(sum(context2, 3), H, B)
        context = context3[:, 1:decbs[i]] # remaining part is not used

        # prediction layer
        out = wy * vcat(context, h_dec) .+ by
        push!(outs, out)
    end
    return outs
end


function loss(model, alldata)
    
    
end
