package io.kinference.completion.suggest

import io.kinference.completion.generating.GenerationInfo
import kotlin.math.roundToInt

class OneVariantFeatures(val context: String, completion: Pair<String, GenerationInfo>, val prefix: String = "") {

    companion object {
        const val prob = "prob"
        const val probs = "probs"
        const val mean_log_prob = "mean_log_prob"
        const val match_prefix = "match_prefix"
        const val prefix_matched_cnt = "prefix_matched_cnt"
        const val first_token_prob = "first_token_prob"
        const val max_prob = "max_prob"
        const val min_prob = "min_prob"
        const val tokens_len = "tokens_len"
        const val word_len = "word_len"
        const val symbol_len = "symbol_len"
        const val alphas_ration = "alphas_ration"
        const val context_len = "context_len"
        const val start_from_word = "start_from_word"
        const val is_repetition = "is_repetition"
    }

    private val completion = completion.first
    private val generation_info = completion.second
    private val features: MutableMap<String, Double> = HashMap()

    init {
        features[prob] = completion.second.probs.foldRight(1.0) { x, y -> x * y }
        features[mean_log_prob] = 1.0
    }

//    self.values = OrderedDict(
//    {OneVariantFeatures.prob: np.prod(self.generation_info.probs()),
//        OneVariantFeatures.probs: self.generation_info.probs().copy(),
//        OneVariantFeatures.mean_log_prob: avg_log_prob(self.generation_info.probs()),
//        OneVariantFeatures.match_prefix: prefix_matched_count(self.prefix, self.completion) == len(self.prefix),
//        OneVariantFeatures.prefix_matched_cnt: prefix_matched_count(self.prefix, self.completion),
//        OneVariantFeatures.first_token_prob: self.generation_info.probs()[0],
//        OneVariantFeatures.max_prob: max(self.generation_info.probs()),
//        OneVariantFeatures.min_prob: min(self.generation_info.probs()),
//        OneVariantFeatures.tokens_len: len(self.generation_info.probs()),
//        OneVariantFeatures.word_len: self.generation_info.word_len,
//        OneVariantFeatures.symbol_len: len(self.completion) - len(prefix),
//        OneVariantFeatures.alphas_ration: sum(c.isalpha() for c in self.completion) / len(self.completion),
//        OneVariantFeatures.context_len: len(self.context),
//        OneVariantFeatures.start_from_word: self.completion[0] == ' ' and self.completion[1].isalpha(),
//        OneVariantFeatures.is_repetition: is_repetition(self.completion, context)
//    }
//    )
}

class Features {
    companion object {
        fun prob(generationInfo: GenerationInfo): Double {
            var prob = 1.0
            generationInfo.probs.forEach {
                prob *= it
            }
            return prob
        }

        fun meanProb(generationInfo: GenerationInfo): Double {
            var probsSum = 0.0
            generationInfo.probs.forEach {
                probsSum += it
            }
            return probsSum / generationInfo.probs.size
        }

        fun firstProb(generationInfo: GenerationInfo): Double {
            return generationInfo.probs[0]
        }

        fun prefixMatchedCount(prefix: String, completion: String): Int {
            for (i in prefix.indices) {
                if (i == completion.length || prefix[i] != completion[i]) {
                    return i
                }
            }
            return prefix.length
        }
    }
}
