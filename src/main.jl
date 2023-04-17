if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import Tokeniser
import GGMLParser

## This is just to remind myself of the API. If I'm not tuning or otherwise
## retraining, is there any point saving the data to yet another file?
import BSON

function save(model, fname)
    BSON.@save(fname, model)
end

function load(fname)
    BSON.@load(fname, model)
    return model
end

files = [
    "../models/13B/ggml-model-f16.bin",
    "../models/13B/ggml-model-f16.bin.1"
]

l = GGMLParser.readmodel(files)
