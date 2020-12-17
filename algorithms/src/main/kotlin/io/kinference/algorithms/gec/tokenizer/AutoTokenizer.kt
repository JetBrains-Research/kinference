package io.kinference.algorithms.gec.tokenizer

import java.nio.file.Paths

/**
 * AutoTokenizer class helper for loading Tokenizers
 */

class AutoTokenizer {
    companion object {
        fun fromPretrained(model_name_or_path: String) : PreTrainedTokenizer{
            if (model_name_or_path == "bert-base-uncased"){
                return BertTokenizer(vocabPath = Paths.get("/Users/Ivan.Dolgov/ivandolgov/projects/vocabs/bert-base-uncased"))
            }
            else{
                throw NotImplementedError("Not implemented yet")
            }
        }
    }
}
