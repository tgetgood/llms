#include <vector>
#include <iostream>
#include <iterator>
#include "sentencepiece_processor.h"
#include "jlcxx.hpp"

sentencepiece::SentencePieceProcessor* sp;
// TODO: check this flag and panic. Maybe we can just check if sp is null, but I
// don't know if that's safe.
bool ready = false;

// TODO: make sure we can unload and reinit without killing the wrapping
// program.
// TODO: look into SetVocabulary and SetEncodeExtraOptions
void init(std::string model) {
  sp = new sentencepiece::SentencePieceProcessor();
  sp->LoadOrDie(model);
  ready = true;
}

// REVIEW: The lifetime of the tokeniser is the full length of the program, but
// we really should clean up. The problem is I don't know C++...
void close() {
  ready = false;
  delete(sp);
}

std::vector<int> encodeIds(std::string text) {
  std::vector<int> ids;
  sp->Encode(text, &ids);
  return ids;
}

std::vector<std::string> encodeStrings(std::string text) {
  std::vector<std::string> tokens;
  sp->Encode(text, &tokens);
  return tokens;
}

// REVIEW: CxxWrap isn't liking overloaded functions. That's probably just my
// lack of knowledge. Can be fixed, but this is good enough for now.
std::string decodeIds(std::vector<int> ids) {
  std::string text;
  sp->Decode(ids, &text);
  return text;
}

std::string decodeStrings(std::vector<std::string> tokens) {
  std::string text;
  sp->Decode(tokens, &text);
  return text;
}

// }

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
  mod.method("init", &init);
  mod.method("encodeIds", &encodeIds);
  mod.method("encodeStrings", &encodeStrings);
  mod.method("decodeIds", &decodeIds);
  mod.method("decodeStrings", &decodeStrings);
}
