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
import GGMLParser
import Lazy

# Dev imports
import BenchmarkTools

# TODO: Take this as input
# REVIEW: Do we actually need to set this? Why not just allow the model to spew
# text until we get sick of it, or run out of memory?
const maxcontext = 512


files = [
    "../models/7B/ggml-model-f16.bin",
    # "../models/13B/ggml-model-f16.bin.1"
]

"""
Applies the root mean square norm from Zhang, Sennrich (2019) along dim.
"""
function RMSNorm(input, gain, dims=1)::Matrix{Float32}
    sqn = size(input)[dims]^(1/2)

    return gain .* input .* sqn ./ sum(x -> x^2, input, dims=dims).^(1/2)
end

"""
Given a real NxM matrix, views each column as alternative real and complex parts
and returns a complex matrix of size Nx(M/2).
"""
function complexify(x, dims=2)
    @assert size(x)[2] % 2 == 0
    return Lazy.@as m x begin
        reshape(m, (
            size(m)[begin:dims-1]...,
            2,
            div(size(m)[dims], 2),
            size(m)[dims+1:end]...
        ))
        mapslices(x -> Complex{eltype(x)}(x[1], x[2]), m, dims=dims)
        reshape(m, (
            size(m)[begin:dims-1]...,
            size(m)[dims+1:end]...
        ))
    end
end

"""
Converts complexified arrays back into their original form.
"""
function decomplexify(x, dims=2)
    return Lazy.@as m x begin
        reshape(m, (
            size(m)[begin:dims-1]...,
            1,
            size(m)[dims:end]...
        ))
        mapslices(x -> [x[1].re, x[1].im], m, dims=dims)
        reshape(m, (
            size(x)[begin:dims-1]...,
            2*size(x)[dims],
            size(x)[dims+1:end]...
        ))
    end
end

"""
Returns the vector of rotations à la Su et al. 2022, in the complex format used
in the llama codebase.
"""
function ropevector(
    T::Type,
    hp::GGMLParser.HyperParameters,
    size::Number,
    θ::Number = 10000
)
    d = hp.embedding/hp.attentionheads
    @assert d % 2 == 0
    θs = map(i -> θ^(-2*i/d), 0:2:d-1)
    ns = 1:size
    base = map(θ -> Complex{T}(cos(θ), sin(θ)), ns' .* θs)

    return repeat(base, hp.attentionheads)
end

function ropevector(d, context, θ = 10000)
    ropevector(Float32, d, context, θ)
end

function attention(input, (; norm, Q, K, V, O), rotv, hparams)
    x = RMSNorm(input, norm)
    #x = decomplexify(complexify(i)*rotv)

    n = div(hparams.embedding, hparams.attentionheads)
    xq = decomplexify(complexify(x*Q)*rotv)
    xk = decomplexify(complexify(x*K)*rotv)

    return input + Flux.NNlib.softmax(xq*xk'/sqrt(n))*(x*V)*O
end

function ffn(input, (; norm, W, W2, V))
    normalised = RMSNorm(input, norm)

    return input + (NNlib.sigmoid.(normalised * W) .* (normalised * V)) * W2
end

function runlayer(rotv, hparams)
    return function(input, layer)
        # TODO: move these to the gpu and figure out a way to load the next layer
        # into vram while this fn is running...
        input = input |> Flux.cpu
        layer = layer |> Flux.cpu

        input = input + attention(input, layer.attention, rotv, hparams)
        input = input + ffn(input, layer.ffn)
        return input
    end
end

function run(embedding, model)
    l = size(embedding)[1]
    hp = model.hyperparameters
    rotv = ropevector(eltype(embedding), hp, l)
    result = reduce(
        runlayer(rotv, model.hyperparameters),
        model.layers,
        init=embedding
    )
    return RMSNorm(result, model.norm) * model.output

end

function padding(input::AbstractArray{T}) where {T}
    return zeros(T, (1, size(input)[2:end]...))
end

function pad(input::AbstractArray{T}) where {T}
    return vcat(input, padding(input))
end

function embedtoken(token, model)
    return model.token_embedding[:, token]
end

function embedtokens(tokens::Vector, model)
    Lazy.@as ts tokens begin
        map(x -> embedtoken(x, model)', ts)
        reduce(vcat, ts)
    end
end

function sampletoken(v)
    # REVIEW: sample with temperature, or stick with frozen?
    # probs = Flux.NNlib.softmax(v ./ 0.8)

    return argmax(v)
end

function decoderloop(prompt, model)
    tokens = Tokeniser.encodeids(prompt)

    for i in 1:1
        embedded = pad(embedtokens(tokens, model))
        next = sampletoken(run(embedded, model)[end, :])

        append!(tokens, next)
    end

    return Tokeniser.decode(tokens)
end

# Prompts generated from llama.cpp with temp 0 for testing compatibility
testprompts = Dict(
    "once upon a time " => "once upon a time 2013-2014\nOnce Upon A Time Season 3 Episode 17 \"Heart Of Gold\"",
    "and bob's your uncle!" => "and bob's your uncle!\nI'm not sure if I've ever mentioned this before, but I have a thing for the word ̈̈\"bob\".",
    "6 tips for underperforming yeti trackers:" => "6 tips for underperforming yeti trackers:\n1. Don’t be afraid to ask questions. If you don’t know what something is, or how it works, just ask!"
)
