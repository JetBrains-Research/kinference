package io.kinference.algorithms.gec.tokenizer

import java.nio.file.Paths

/**
 * AutoTokenizer class helper for loading Tokenizers
 */
class AutoTokenizer {
    companion object {
        fun fromPretrained(modelPath: String): PreTrainedTokenizer {
            return BertTokenizer(vocabPath = Paths.get(modelPath))
        }
    }
}
