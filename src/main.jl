if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import Flux
import Flux.NNlib as NNlib
import OneHotArrays
import Tokeniser
import ModelLoader

files = [
    "../models/7B/ggml-model-f16.bin",
    # "../models/13B/ggml-model-f16.bin.1"
]

"""
Applies the root mean square norm from Zhang, Sennrich (2019) along dim.
"""
function RMSNorm(input, gain, dim=1)
    sqn = size(input)[dim]^(1/2)

    return gain .* input .* sqn ./ sum(x -> x^2, input, dims=dim).^(1/2)
end

function attention(input, (; norm, Q, K, V, O), rotv)
    normalised = RMSNorm(input, norm)



end

function ffn(input, (; norm, W, W2, V))
    normalised = RMSNorm(input, norm)

    return (NNlib.sigmoid.(normalised * W) .* (normalised * V)) * W2
end

##### rough sketch of execution
function runlayer(input, layer)
    input = input .+ attention(input, layer.attention)
    input = input .+ ffn(input, layer.ffn)
    return input
end

function run(input, model)
    # REVIEW: We want to load the layers to the gpu as we go. How do we overlap
    # the loading of future layers with the execution of the current ones to
    # minimise latency?
    #
    # The whole thing *will not* fit in vram at once, but ~10 layers will no
    # problem.
    embedding = embed(input, model.token_embeddings)
    normalised = RMSNorm(embedding, norm)
    return reduce(runlayer, model.layers, init=normalised) * model.output
end

function textinput(text::String, model)
    tokens = Tokeniser.encode(text)

    ids = map(x -> get(model.tokens.byid, x, 1), tokens)

    return OneHotArrays.onehotbatch(ids, 1:model.hyperparameters.vocabsize)
end

"""
Returns naive closest match instead of sampling from the token distribution.
"""
function simpletextoutput(vs, model)
    ids = OneHotArrays.onecold(vs, 1:model.hyperparameters.vocabsize)
    tokens = map(x -> model.tokens.tokens[x].token, ids)
    return Tokeniser.decode(tokens)
end
