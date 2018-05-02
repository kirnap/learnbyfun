# To preprocess image files


mutable struct PoImg
    vivec::Vector{Float32} # Visual Representation
    cid::Int               # Category ID
    name::String           # Name of Image
end

mutable struct ImSeq
    members::Array{PoImg}
end

# To get rid of ugly show
Base.show(io::IO, p::PoImg)=print(io,p.cid, " ", p.name)
Base.length(p::PoImg)=length(p.vivec)

function readdata(jsonfile, resmodel)
    dataset = []
    bigdata = JSON.parsefile(jsonfile) # Array of any
    info("json file has a length of $(length(bigdata))")
    for data in bigdata #
        impath = joinpath("data/images", data["set_id"])
        imgs = data["items"] # retrive original image
        (length(imgs) < 3 && continue)
        img_seq = []
        for i in imgs
            ((i["name"] == "polyvore" || i["name"] == "") && continue)
            (length(img_seq) > 8 && break); # Todo: we may need more
            cid = i["index"]; tp = joinpath(impath, string(cid, ".jpg"))
            vivec = get_imvec(tp, resmodel)
            (vivec == nothing) && continue
            push!(img_seq, (PoImg(vivec, i["categoryid"], i["name"])))
        end
        (length(dataset) % 1000 == 0) && (@show length(dataset))
        push!(dataset, img_seq)
    end
    return dataset
end


# Takes a url or file name and returns an Array{RGB} 
# It does not resize or modify the original image. 
function loadimage(img)
    global _imgcache
    if contains(img,"://")
        info("Downloading $img")
        a0 = load(download(img))
    else
        a0 = load(img)
    end
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b10 = a1[i1+1:i1+224,j1+1:j1+224]
    return b10
end


# It takes output of loadimage and 
# modifies it by resizing and subtracting average image 
# provided my resnet file 
function prepare_img(img,averageImage)
    # ad-hoc solution for Mac-OS image
    macfix = nothing
    try
        macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(img))
    catch
        return nothing
    end
    c1 = permutedims(macfix, (3,2,1))
    d1 = convert(Array{Float32}, c1)
    e1 = reshape(d1[:,:,1:3], (224,224,3,1))
    f1 = (255 * e1 .- averageImage) 
    g1 = permutedims(f1, [2,1,3,4])
    g2 = (gpu() >= 0 ? KnetArray{Float32}(g1) : Array{Float32}(g1))
    return g2
end


# extracts the image vector from given resnet
function get_imvec(impath, resmodel)
    resms, resws, resfunc, averageImage = resmodel  
    input = prepare_img(loadimage(impath), averageImage)
    (input == nothing) && (return nothing)
    return Array(vec(resfunc(resws, input, resms)))
end

# creates the bundle to use for ResNet
function create_resmodel()
    resname = "imagenet-resnet-50-dag"
    info("Resmodel is being loaded")
    model = load_model(resname)
    avgimg = model["meta"]["normalization"]["averageImage"]
    avgimg  = convert(Array{Float32}, avgimg)
    atype = (gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
    resws, resms = get_params(model["params"], atype);
    resfunc = resnet50
    resmodel = (resms, resws, resfunc, avgimg)
    return resmodel
end
