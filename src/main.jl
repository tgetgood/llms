if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

# import .Tokeniser

##### Structs reverse engineered from ggml binary format.

struct VersionInfo
    magic::UInt32
    version::UInt32
end

struct HyperParameters
    vocabsize::UInt32
    embeddings::UInt32
    swiglu_multiple::UInt32
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

struct Layer{T}
    name::String
    data::T
end

##### Wrapper to read a split file set as if it were one file

mutable struct FilesReader
    const filenames::Vector{String}
    index::UInt
    stream::IOStream
end

function reader(fnames)
    return FilesReader(fnames, 1, open(fnames[1]))
end

function Base.skip(h::FilesReader, count)
    skip(h.stream, count)
end

function Base.read!(handle::FilesReader, buff)
    if eof(handle.stream)
        close(handle.stream)
        if handle.index <= length(handle.filenames)
            handle.index += 1
            handle.stream = open(handle.filenames[handle.index])
        else
            throw(Base.EOFError)
        end
    end

    read!(handle.stream, buff)
end

function Base.close(handle::FilesReader)
    close(handle.stream)
end

function Base.eof(h::FilesReader)
    return h.index == length(h.filenames) && eof(h.stream)
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

function readarray(handle, rank)
    # FIXME: hardcoded Float16
    layer = Array{Float16}(undef, rank...)
    read!(handle, layer)
    return layer
end

##### Attempt at reading

function n_ff((; swiglu_multiple, embeddings))::Int
    return swiglu_multiple *
        floor((2*(4*embeddings)/3 + swiglu_multiple - 1)/swiglu_multiple)
end

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
    (; vocabsize, embeddings, layers) = hparams
    f16size = sizeof(Float16)
    f32size = sizeof(Float32)


    # TODO: A lisp style "align the comments column" formatter would be handy.
    return embeddings * vocabsize * f16size + # Token embeddings
        embeddings * f32size + # Norm
        embeddings * vocabsize * f16size + # output buffer
        layers * embeddings * f32size  + # attention norm
        layers * embeddings^2 * f16size * 4 + # wq, wk, wv, wo
        layers * embeddings * f32size + # ffn norm # TODO: ffn?
        layers * n_ff(hparams) * embeddings * f16size * 3 + # w1, w2, w3
        contextwidth * layers * embeddings * f32size * 2 # memory k & v
end

function readlayer(model)
    lm = readbits(model, LayerMeta)

    rank = Vector{UInt32}(undef, lm.dimensions)
    read!(model, rank)

    name = readstring(model, lm.namelength)

    data = readarray(model, rank)

    return Layer(name, data)
end

function readlayermeta(model)
    lm = readbits(model, LayerMeta)

    rank = Vector{UInt32}(undef, lm.dimensions)
    read!(model, rank)

    name = readstring(model, lm.namelength)

    datasize = reduce(*, rank, init=sizeof(Float16))

    skip(model, datasize)

    return (name, rank, datasize)
end

function readmodel(files)
    model = reader(files)

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
        l = readlayermeta(model)
        println(l)
       push!(layers, l)
    end
    return layers
end

##### Scratch code.

files = ["../models/7B/ggml-model-f16.bin"]

l = readmodel(files)
