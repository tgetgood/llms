if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import Tokeniser

##### Load model from ggml binary. TODO: Refactor into module, make it a script.

struct VersionInfo
    magic::UInt32
    version::UInt32
end

# Would it be better to create a new "bits" type for this?
struct LlamaParams
    vocabsize::UInt32
    embeddings::UInt32
    mults::UInt32
    attentionheads::UInt32
    layers::UInt32
    rots::UInt32
    f16::UInt32
end

#FIXME: const
datapartitions = Dict(
    4096 => 1,
    5120 => 2,
    6656 => 3,
    8192 => 8
)

function readbin(handle, type)
    temp = Vector{type}(undef, 1)
    read!(handle, temp)
    return temp[1]
end

function readstring(handle, bytes)
    temp = Vector{UInt8}(undef, bytes)
    read!(handle, temp)
    return String(temp)
end

model = open("../models/7B/ggml-model-f16.bin")

version = readbin(model, VersionInfo)

@assert version.magic == 0x67676d66 && version.version == 1 "This script can only parse ggml version 1"

hparams = readbin(model, LlamaParams)

# Maximum number of tokens to consider
contextlength = 512

# How many files is this broken into?
files = datapartitions[hparams.embeddings]

# What does this number signify?
n_ff::Int = floor(
    (2*(4*hparams.embeddings)/3 + hparams.mults - 1)/hparams.mults
) * hparams.mults

struct ScoredToken
    token::String
    score::Float32
    id::UInt32
end

# Vector of ScoredTokens, where index matched id field.
tokens = Vector{ScoredToken}(undef, hparams.vocabsize)

idsbytoken = Dict{String, UInt32}()

for i in 1:hparams.vocabsize
    length = readbin(model, UInt32)

    token = ScoredToken(
        readstring(model, length),
        readbin(model, Float32),
        i
    )

    tokens[i] = token
    idsbytoken[token.token] = i
end
