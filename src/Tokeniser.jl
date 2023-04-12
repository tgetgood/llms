module Tokeniser

module SentencePieceWrapper
using CxxWrap
const splib = "../sentencepiece/shared/spshim.so"

@wrapmodule(splib)

function __init__()
    @initcxx
end
end

using CxxWrap
import .SentencePieceWrapper
const modelpath = "../models/tokenizer.model"

function __init__()
    SentencePieceWrapper.init(modelpath)
end

function encodeids(text::String)::Vector{Int32}
    return SentencePieceWrapper.encodeIds(text)
end

function encode(text::String)::Vector{String}
    return SentencePieceWrapper.encodeStrings(text)
end

function decode(tokens::Vector{Int32})::String
    return SentencePieceWrapper.decodeIds(
        CxxWrap.StdLib.StdVector(tokens)
    )
end

function decode(tokens::Vector{String})::String
    return SentencePieceWrapper.decodeStrings(
        CxxWrap.StdLib.StdVector(
            map(CxxWrap.StdLib.StdString, tokens)
        )
    )
    #FIXME: This should work just as well:
    # join(tokens)
end
end
