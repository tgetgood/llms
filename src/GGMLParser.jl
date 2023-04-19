module GGMLParser

##### Structs reverse engineered from ggml binary format.

struct VersionInfo
    magic::UInt32
    version::UInt32
end

struct HyperParameters
    vocabsize::UInt32
    embedding::UInt32
    ffn_inner_mulitple::UInt32
    attentionheads::UInt32
    layers::UInt32
    rots::UInt32
    f16::UInt32
end

### Token dictionary

struct ScoredToken
    token::String
    score::Float32
    id::UInt32
end

### Transformer layers

struct LayerMeta
    dimensions::UInt32
    namelength::UInt32
    ftype::UInt32 # 1 => Float16 values (mostly)
end

##### General binary reading helpers

function readbits(handle, type)
    temp = Vector{type}(undef, 1)
    read!(handle, temp)
    return temp[1]
end

function readstring(handle, bytes)
    temp = Vector{UInt8}(undef, bytes)
    read!(handle, temp)
    return String(temp)
end

##### Attempt at reading

function readfilemetadata(model)

    version = readbits(model, VersionInfo)

    @assert(
        version.magic == 0x67676d66 && version.version == 1,
        "This script can only parse ggmf version 1 (corresponds to LLAMA_FILE_VERSION_GGMF_V1 in llama.cpp)."
    )

    hparams = readbits(model, HyperParameters)

    @assert hparams.f16 == 1 "Currently only supporting Float16."

    return (version, hparams)
end

function ffn_inner_units((; ffn_inner_mulitple, embedding))::Int
    return ffn_inner_mulitple *
        floor((2*(4*embedding)/3 + ffn_inner_mulitple - 1)/ffn_inner_mulitple)
end


function readtokendict(model, (; vocabsize))
    # Vector of ScoredTokens, where index matched is field.
    tokens = Vector{ScoredToken}(undef, vocabsize)

    idsbytoken = Dict{String, UInt32}()

    for i in 1:vocabsize
        length = readbits(model, UInt32)

        token = ScoredToken(
            readstring(model, length),
            readbits(model, Float32),
            i
        )

        tokens[i] = token
        idsbytoken[token.token] = i
    end
    return (tokens, idsbytoken)
end

# Memory required to load the model (not counting collection overhead) in bytes
function modelsize(hparams, contextwidth)
    (; vocabsize, embedding, layers) = hparams
    f16size = sizeof(Float16)
    f32size = sizeof(Float32)


    # TODO: A lisp style "align the comments column" formatter would be handy.
    return embedding * vocabsize * f16size + # Token embedding
        embedding * f32size + # Norm
        embedding * vocabsize * f16size + # output buffer
        layers * embedding * f32size  + # attention norm
        layers * embedding^2 * f16size * 4 + # wq, wk, wv, wo
        layers * embedding * f32size + # ffn norm # TODO: ffn?
        layers * ffn_inner_units(hparams) * embedding * f16size * 3 + # w1, w2, w3
        contextwidth * layers * embedding * f32size * 2 # memory k & v
end

# FIXME: const
f32layers = Set([
    "norm.weight"
    "layers.0.attention_norm.weight"
])

# FIXME: const
columnlayers = Set([
 "tok_embeddings.weight"
])

function eltypelookup(name)
    if endswith(name, "norm.weight")
        return Float32
    else
        #FIXME: hardcoded default element type
        return Float16
    end
end

function readlayer(model)
    lm = readbits(model, LayerMeta)

    rank = Vector{UInt32}(undef, lm.dimensions)
    read!(model, rank)

    if lm.namelength > 100
        throw("name is too long, this indicates a parsing error.")
    end

    name = readstring(model, lm.namelength)
    eltype = eltypelookup(name)

    data = Array{eltype}(undef, rank...)
    read!(model, data)

    return (name, data)
end

function readlayermeta(model)
    lm = readbits(model, LayerMeta)

    rank = Vector{UInt32}(undef, lm.dimensions)
    read!(model, rank)

    if lm.namelength > 100
        throw("name is too long, this indicates a parsing error.")
    end

    name = readstring(model, lm.namelength)
    eltype = eltypelookup(name)

    datasize = reduce(*, rank, init=sizeof(eltype))

    skip(model, datasize)

    return (name, rank, datasize)
end

function readfile(file)
    model = open(file)

    # Maximum number of tokens to consider
    # FIXME: Presumably this should not be hardcoded...
    contextwidth = 512

    (_, hparams) = readfilemetadata(model)
    (tokens, idsbytoken) = readtokendict(model, hparams)

    @assert(
        # is the extra 100MB reasonable? sufficient? I'm just making this up.
        modelsize(hparams, contextwidth) + 100*2^20 < Sys.free_memory(),
        "Not enough memory to load model, aborting."
    )

    layers = []
    while !eof(model)
        push!(layers, readlayer(model))
    end
    return (hparams, layers)
end

function mergelayer(a, b)
end

function mergeshards(layers)
    if length(layers) == 1
        return layers[1]
    else
        return reduce(mergelayer, layers[2:end], init=layers[1])
    end
end

"""
Reads the model stored in the list of `files` passed and returns a tuple
containing hyperparameters and a list of (name, tensor) pairs.

N.B. If the files are not in order, the matricies will be incorrect.
"""
function readmodel(files)
    layers = []
    hparams = undef
    for file in files
        (h, l) = readfile(file)
        if hparams == undef
            hparams = h
        else
            @assert(
            h == hparams,
                "file metadata mismatch! got " * string(h) *
                " but expected " * stirng(hparams) *
                ". Aborting."
            )
        end
        push!(layers, l)
    end

    # TODO: layers are split across files, which is a bizzare way of dealing
    # with it, but so be it.
    return (hparams, mergeshards(layers))
end

end
