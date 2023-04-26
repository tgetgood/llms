module Tokeniser

module SentencePieceWrapper
using CxxWrap
const splib = get(ENV, "SP_SHIM_PATH", "../sentencepiece") * "/shared/spshim.so"

@wrapmodule(splib)

function __init__()
    @initcxx
end
end

using CxxWrap
import .SentencePieceWrapper
const modelpath = get(ENV, "TOKENISER_MODEL", "../models/tokenizer.model")

"""
Converts tokens to use spaces as in llama instead of sentencepiece's
underscore(U+2581).
"""
function normalise(token::String)
    return replace(token, "▁" => " ")
end

function denormalise(token::String)
    return replace(token, " " => "▁")
end

function __init__()
    SentencePieceWrapper.init(modelpath)
end

function encodeids(text::String)::Vector{Int32}
    return SentencePieceWrapper.encodeIds(text)
end

# FIXME: There's some conversion that works for vectors of
# CxxWrap.StdLib.StdStringDereferenced to vectors of strings, but not for the
# elements directly, so I'm calling map inside a separate function call.
#
# I need to learn how the type conversion / inference system works. There's
# probably a more elegant and general fix for this
function encoderaw(text::String)::Vector{String}
    return SentencePieceWrapper.encodeStrings(text)
end

function encode(text::String)::Vector{String}
    return map(normalise, encoderaw(text))
end

function decode(tokens::Vector{Int32})::String
    return SentencePieceWrapper.decodeIds(
        CxxWrap.StdLib.StdVector(tokens)
    )
end

function decode(tokens::Vector{String})::String
    return SentencePieceWrapper.decodeStrings(
        CxxWrap.StdLib.StdVector(
            map(CxxWrap.StdLib.StdString ∘ denormalise, tokens)
        )
    )
    #FIXME: This should work just as well, but needs testing
    # join(tokens)
end
end
