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

# TODO: Take this as input
const contextwidth = 512


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
function ropevector(T::Type, d::Number, context::Number, θ::Number = 10000)
    @assert d % 2 == 0
    θs = map(i -> θ^(-2*i/d), 0:2:d-1)
    ns = 1:context

    return map(θ -> Complex{T}(cos(θ), sin(θ)), ns' .* θs)
end

function ropevector(d, context, θ = 10000)
    ropevector(Float32, d, context, θ)
end

function attention(input, (; norm, Q, K, V, O), rotv, hparams)
    x = RMSNorm(input, norm)
    #x = decomplexify(complexify(i)*rotv)

    n = div(hparams.embedding, hparams.attentionheads)
    r = repeat(rotv, hparams.attentionheads)
    xq = decomplexify(complexify(x*Q)*r)
    xk = decomplexify(complexify(x*K)*r)

    return Flux.NNlib.softmax(xq*xk'/sqrt(n))*(x*V)*O'
end

function ffn(input, (; norm, W, W2, V))
    normalised = RMSNorm(input, norm)

    return (NNlib.sigmoid.(normalised * W) .* (normalised * V)) * W2
end

##### rough sketch of execution
function runlayer(rotv, hparams)
    return function(input, layer)
        # TODO: move these to the gpu and figure out a way to load the next layer
        # into vram while this fn is running...
        input = input |> Flux.cpu
        layer = layer |> Flux.cpu

        input = input .+ attention(input, layer.attention, rotv, hparams)
        input = input .+ ffn(input, layer.ffn)
        return input
    end
end

function run(input, model)
    hp = model.hyperparameters
    embedding = pad(embedtext(input, model), contextwidth)
    normalised = RMSNorm(embedding, model.norm)
    rotv = ropevector(
        eltype(embedding),
        div(hp.embedding, hp.attentionheads),
        contextwidth
    )
    result = reduce(
        runlayer(rotv, model.hyperparameters),
        model.layers,
        init=normalised
    )
    return result * model.output
end

function pad(input::AbstractArray{T}, n::Int) where {T}
    padding = zeros(T, (n-size(input)[1], size(input)[2:end]...))
    return vcat(input, padding)
end

function embedtext(text::String, model)
    return transpose(
    Lazy.@>> Tokeniser.encode(text) begin
        map(x -> get(model.tokens.idmap, x, 1))
        map(x -> model.token_embedding[:, x])
        reduce(hcat)
    end
    )
end

"""
Returns naive closest match instead of sampling from the token distribution.
"""
function simpletextoutput(vs, model)
    @info size(vs) model.hyperparameters.vocabsize
    ids = OneHotArrays.onecold(vs', 1:model.hyperparameters.vocabsize)
    tokens = map(x -> model.tokens.tokens[x].token, ids)
    return Tokeniser.decode(tokens)
end
