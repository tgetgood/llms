if !("./" in LOAD_PATH)
    push!(LOAD_PATH, "./")
end

# import Tokeniser
import ModelLoader

files = [
    "../models/7B/ggml-model-f16.bin",
    # "../models/13B/ggml-model-f16.bin.1"
]



##### rough sketch of execution
function runlayer(x, layer)
    x = x + attention(attnorm(x))
    x = x + ffn(ffnnorm(x))
    return x
end

function run(x, layers)
    return unembed(reduce(runlayer, layers, init=embed(x)))
end
