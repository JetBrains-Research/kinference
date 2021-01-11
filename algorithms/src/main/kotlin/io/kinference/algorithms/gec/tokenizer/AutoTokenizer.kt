package io.kinference.algorithms.gec.tokenizer

import java.nio.file.Paths

/**
 * AutoTokenizer class helper for loading Tokenizers
 */
class AutoTokenizer {
    companion object {
        fun fromPretrained(modelNameOrPath: String): PreTrainedTokenizer {
            if (modelNameOrPath == "bert-base-uncased") {
                return BertTokenizer(vocabPath = Paths.get("/Users/Ivan.Dolgov/ivandolgov/projects/vocabs/bert-base-uncased"))
            } else {
                throw NotImplementedError("Not implemented yet")
            }
        }
    }
}
