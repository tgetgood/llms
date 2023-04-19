if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import Tokeniser
import GGMLParser

files = [
    "../models/7B/ggml-model-f16.bin",
    # "../models/13B/ggml-model-f16.bin.1"
]

struct Attention{T}
    Q::Array{T}
    K::Array{T}
    V::Array{T}
    O::Array{T}
    norm::Vector{Float32}
end

struct FeedForward{T}
    W::Array{T}
    W2::Array{T}
    V::Array{T}
    norm::Vector{Float32}
end

struct Layer{T}
    attention::Attention{T}
    ff::FeedForward{T}
end

struct Model{T}
    token_embeddings::Array{T}
    norm::Vector{Float32}
    output::Array{T}
    layers::Vector{Layer{T}}
end

function injestggml(files)
    l = GGMLParser.readmodel(files)
    n = l[1].layers
    layers::Vector{Layer{Float16}} = []
    layer = 0
    i = 4
    d = Dict()
    while i <= length(l[2])
        m = match(Regex("\\."*string(layer)*"\\.(.*)"), l[2][i][1])
        if m != nothing
            d[m[1]] = l[2][i][2]
            i = i + 1
        else
            temp = Layer(
                Attention(
                    d["attention.wq.weight"],
                    d["attention.wk.weight"],
                    d["attention.wv.weight"],
                    d["attention.wo.weight"],
                    d["attention_norm.weight"]
                ),
                FeedForward(
                    d["feed_forward.w1.weight"],
                    d["feed_forward.w2.weight"],
                    d["feed_forward.w3.weight"],
                    d["ffn_norm.weight"]
                )
            )
            push!(layers, temp)
            layer = layer + 1
            d = Dict()
        end
    end
    return Model(
        l[2][1][2],
        l[2][2][2],
        l[2][3][2],
        layers
    )
end


##### rough sketch of execution
function runlayer(x, layer)
    x = x + attention(attnorm(x))
    x = x + ffn(ffnnorm(x))
    return x
end

function run(x, layers)
    return unembed(reduce(runlayer, layers, init=embed(x)))
end
