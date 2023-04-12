# This is intentionally non executable. `source` it, don't run it. 

# These are system dependent, especially the JLCXX headers which will move if the hash changes
SENTENCEPIECE_HEADERS_PATH=~/projects/reference/sentencepiece/src/
JLCXX_HEADERS_PATH=$(julia -e 'import Pkg; Pkg.activate(".."); import CxxWrap; println(CxxWrap.prefix_path());')/include/jlcxx/

export CPATH=$CPATH:$SENTENCEPIECE_HEADERS_PATH:$JLCXX_HEADERS_PATH
