package io.kinference.algorithms.completion.ranking

import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.CompletionModels
import io.kinference.algorithms.completion.generation.GenerationInfo
import io.kinference.algorithms.completion.suggest.ranking.WordTrieIterativeGolfRanking
import io.kinference.algorithms.completion.tokenizer.BPETokenizer
import org.junit.jupiter.api.*

class WordTrieIterativeGolfRankingTest {
    @Test
    @Tag("heavy")
    fun testGolfTrie() {
        val tokenizer = BPETokenizer(CompletionModels.v4.loader)
        val completions = mutableListOf(
            CompletionModel.CompletionResult(
                " not the first time that", GenerationInfo(
                listOf(0.08889821171760559, 0.08719435334205627, 0.08893933892250061, 0.8582536578178406, 0.331612229347229),
                listOf(407, 262, 717, 640, 326))
            ),
            CompletionModel.CompletionResult(
                " not a good", GenerationInfo(
                listOf(0.08889821171760559, 0.1470089703798294, 0.027259405702352524),
                listOf(407, 257, 922))
            ),
            CompletionModel.CompletionResult(
                " not possible to", GenerationInfo(
                listOf(0.08889821171760559, 0.02475619688630104, 0.6013234257698059),
                listOf(407, 1744, 284))
            ),
            CompletionModel.CompletionResult(
                " not going to be", GenerationInfo(
                listOf(0.08889821171760559, 0.014917194843292236, 0.8694486021995544, 0.24544979631900787),
                listOf(407, 1016, 284, 307))
            ),
            CompletionModel.CompletionResult(
                " not just a", GenerationInfo(
                listOf(0.08889821171760559, 0.023179415613412857, 0.20381753146648407),
                listOf(407, 655, 257))
            ),
            CompletionModel.CompletionResult(
                " not the case that", GenerationInfo(
                listOf(0.08889821171760559, 0.08719435334205627, 0.1029171347618103, 0.27619943022727966),
                listOf(407, 262, 1339, 326))
            ),
            CompletionModel.CompletionResult(
                " not the only", GenerationInfo(
                listOf(0.08889821171760559, 0.08719435334205627, 0.07642960548400879),
                listOf(407, 262, 691))
            ),
        )

        val ranker = WordTrieIterativeGolfRanking(tokenizer, 5, 0.0)
        val topListed = ranker.rank("It is", " not", completions)
        val result = topListed.map { it.text }

        val target = listOf(" not the first time that", " not possible to", " not going to be", " not the case that", " not the only")
        Assertions.assertEquals(target, result)
    }
}
