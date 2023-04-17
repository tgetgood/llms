if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

import .Tokeniser
import .GGMLParser


files = ["../models/7B/ggml-model-f16.bin"]

l = GGMLParser.readmodel(files)
