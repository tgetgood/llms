module ModelLoader

import GGMLParser
import Tokeniser

export ingestggml

struct Attention{T}
    Q::Array{T}
    K::Array{T}
    V::Array{T}
    O::Array{T}
    norm::Array{Float32}
end

struct FeedForward{T}
    W::Array{T}
    W2::Array{T}
    V::Array{T}
    norm::Array{Float32}
end

struct Layer{T}
    attention::Attention{T}
    ffn::FeedForward{T}
end

struct TokenDict
    tokens::Vector{GGMLParser.ScoredToken}
    idmap::Dict{String, UInt32}
end

struct Model{T}
    hyperparameters::GGMLParser.HyperParameters
    tokens::TokenDict
    token_embedding::Array{T}
    norm::Array{Float32}
    output::Array{T}
    layers::Vector{Layer{T}}
end

function normalisetokens(tokens)
    return map(tokens) do (; token, score, id)
        return GGMLParser.ScoredToken(
            Tokeniser.normalise(token),
            score,
            id
        )
    end
end

function ingestggml(T, files)
    (hparams, tokendict, rawlayers) = GGMLParser.readmodel(files)
    n = hparams.layers
    @assert 9 * n + 3 == length(rawlayers) "invalid model metadata"
    layers::Vector{Layer{T}} = []
    layer = 0
    i = 4
    d = Dict()
    while i <= length(rawlayers)
        m = match(Regex("\\."*string(layer)*"\\.(.*)"), rawlayers[i][1])
        if m != nothing
            d[m[1]] = rawlayers[i][2]
            i = i + 1
        else
            an = d["attention_norm.weight"]
            fn = d["ffn_norm.weight"]
            temp = Layer{T}(
                Attention{T}(
                    d["attention.wq.weight"],
                    d["attention.wk.weight"],
                    d["attention.wv.weight"],
                    d["attention.wo.weight"],
                    reshape(an, (1, size(an)...))
                ),
                FeedForward{T}(
                    d["feed_forward.w1.weight"],
                    d["feed_forward.w2.weight"],
                    d["feed_forward.w3.weight"],
                    reshape(fn, (1, size(fn)...))
                )
            )
            push!(layers, temp)
            layer = layer + 1
            d = Dict()
        end
    end
    norm = rawlayers[2][2]
    return Model{T}(
        hparams,
        TokenDict(normalisetokens(tokendict[1]), tokendict[2]),
        rawlayers[1][2],
        reshape(norm, (1, size(norm)...)),
        rawlayers[3][2],
        layers
    )
end

function ingestggml(files)
    return ingestggml(Float16, files)
end


end
