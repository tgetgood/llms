if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import Flux
import Flux.NNlib as NNlib
import OneHotArrays
import SparseArrays
import LinearAlgebra
import Tokeniser
import ModelLoader
import Lazy

# Dev imports
import BenchmarkTools

files = [
    "../models/7B/ggml-model-f16.bin",
    # "../models/13B/ggml-model-f16.bin.1"
]

"""
Applies the root mean square norm from Zhang, Sennrich (2019) along dim.
"""
function RMSNorm(input, gain, dim=1)::Matrix{Float32}
    sqn = size(input)[dim]^(1/2)

    return gain .* input .* sqn ./ sum(x -> x^2, input, dims=dim).^(1/2)
end

function attention(input, (; norm, Q, K, V, O), rotv)
    normalised = RMSNorm(input, norm)

    return normalised
end

function ffn(input, (; norm, W, W2, V))
    normalised = RMSNorm(input, norm)

    return (NNlib.sigmoid.(normalised * W) .* (normalised * V)) * W2
end

##### rough sketch of execution
function runlayer(input, layer)
    glayer = layer
    input = input .+ attention(input, glayer.attention, [])
    input = input .+ ffn(input, glayer.ffn)
    return input
end

function run(input, model)
    # REVIEW: We want to load the layers to the gpu as we go. How do we overlap
    # the loading of future layers with the execution of the current ones to
    # minimise latency?
    #
    # The whole thing *will not* fit in vram at once, but ~10 layers will no
    # problem.
    embedding = quicktextinput(input, model)
    normalised = RMSNorm(embedding, model.norm)
    result = reduce(runlayer, model.layers, init=normalised)
    return (result * (model.output))
end

function textinput(text::String, model)
    tokens = Tokeniser.encode(text)

    ids = map(x -> get(model.tokens.idmap, x, 1), tokens)

    return SparseArrays.SparseMatrixCSC(
        OneHotArrays.onehotbatch(ids, 1:model.hyperparameters.vocabsize)
    )
end

function quicktextinput(text::String, model)
    return transpose(
    Lazy.@>> Tokeniser.encode(text) begin
        map(x -> get(model.tokens.idmap, x, 1))
        map(x -> model.token_embeddings[:, x])
        reduce(hcat)
    end
    )
end

"""
Returns naive closest match instead of sampling from the token distribution.
"""
function simpletextoutput(vs, model)
    ids = OneHotArrays.onecold(vs, 1:model.hyperparameters.vocabsize)
    tokens = map(x -> model.tokens.tokens[x].token, ids)
    return Tokeniser.decode(tokens)
end
