# Training script
using JSON, MAT, Knet, Images, JLD
# to save the model related
import JLD: writeas, readas
import Knet: RNN
type RNNJLD; inputSize; hiddenSize; numLayers; dropout; inputMode; direction; mode; algo; dataType; end
writeas(r::RNN) = RNNJLD(r.inputSize, r.hiddenSize, r.numLayers, r.dropout, r.inputMode, r.direction, r.mode, r.algo, r.dataType)
readas(r::RNNJLD) = rnninit(r.inputSize, r.hiddenSize, numLayers=r.numLayers, dropout=r.dropout, skipInput=(r.inputMode==1), bidirectional=(r.direction==1), rnnType=(:relu,:tanh,:lstm,:gru)[1+r.mode], algo=r.algo, dataType=r.dataType)[1]
type KnetJLD; a::Array; end
writeas(c::KnetArray) = KnetJLD(Array(c))
readas(d::KnetJLD) = (gpu() >= 0 ? KnetArray(d.a) : d.a)

include("conv_model.jl")
include("preprocess.jl")
include("model1.jl")


function main()
    # data
    info("Creating dataset")
    tdata = JLD.load("data/train_data.jld");
    tdata = filter(x->length(x)>2, tdata["dataset"]);
    cset = Set{Int}()
    for item in tdata; map(x->push!(cset, x.cid), item);end;
    csize = maximum(collect(cset)) # for softmax init

    # test related
    testdata = JLD.load("data/test_data.jld", "testset");


    imbatches = minibatch2(;batchsize=1, tdata=tdata, ncols=csize);
    testbatches = minibatch2(;batchsize=1, tdata=testdata, ncols=csize);


    # hyper-parameters and model
    inputdim = 2048; outdim = csize; hidden = 450; embed = 128;
    info("Creating model hidden $hidden")
    model = initmodel(inputdim, embed, hidden, csize)
    optims = optimizers(model, Adam)

    # train
    info("training started...")
    j = length(imbatches)
    prevacc = acc1 = accuracy(model, imbatches)
    println("epoch 0 acc $acc1")
    for epoch in 1:100
        lepoch=0.0
        shuffle!(imbatches)
        for i1 in 1:8:j
            j1 = min(j, i1+7);
            instances = imbatches[i1:j1]
            g1, l1 = bigrad2(model, instances)
            update!(model, g1, optims)
            lepoch += l1
        end
        lval = lepoch / length(imbatches)
        acc2 = accuracy(model, imbatches)
        acc1 = accuracy(model, testbatches)
        if acc1 > prevacc
            mname = string(hidden, "_best_model.jld") #string(epoch, "_model.jld")
            JLD.save(mname, "model", model)
            prevacc = acc1
            println("epoch $epoch lossvall $lval trainacc $acc2 testacc $acc1 $mname saved.")
        else
            println("epoch $epoch loss vall $lval trainacc $acc2 testacc $acc1")
        end

    end
    
    #info("saving mode 100l")
    #JLD.save("end_100model2.jld", "model", model)
end
!isinteractive() && main()


# TODO: interactive test mode implementation
# 1. Read the images and creat resnet embeddings
# 2. Go forwad and get the probabilities, and category-id
# 3. Based on that prediction get a random item/ get the best possible item
