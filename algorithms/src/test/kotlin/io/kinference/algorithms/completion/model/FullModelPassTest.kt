package io.kinference.algorithms.completion.model

import io.kinference.algorithms.completion.CompletionModels
import io.kinference.algorithms.completion.CompletionConfig
import io.kinference.algorithms.completion.CompletionModel
import io.kinference.algorithms.completion.suggest.FairseqCompletionsCollector
import io.kinference.algorithms.completion.suggest.filtering.FilterModel
import io.kinference.algorithms.completion.suggest.filtering.ProbFilterModel
import io.kinference.algorithms.completion.suggest.ranking.FirstProbRankingModel
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.lang.System.currentTimeMillis

class FullModelPassTest {
    @Test
    @Tag("heavy")
    fun test() {
        val completionsCollector = FairseqCompletionsCollector(CompletionModels.v5)
        val rankingModel = FirstProbRankingModel()
        val filterModel = ProbFilterModel()
        val postFilterModel: FilterModel? = null

        val completionModel = CompletionModel(completionsCollector, rankingModel, filterModel, postFilterModel)

//        interaction(completionModel, config)
        speedTest(completionModel, CompletionModels.v5, 30, 20)
    }


    @Test
    @Tag("heavy")
    fun `test bpe problem`() {
        val completionsCollector = FairseqCompletionsCollector(CompletionModels.v5)
        val rankingModel = FirstProbRankingModel()
        val filterModel = ProbFilterModel()
        val postFilterModel: FilterModel? = null

        val completionModel = CompletionModel(completionsCollector, rankingModel, filterModel, postFilterModel)

        completionModel.complete("Hello my dear friends", "", CompletionModels.v5)
    }

    private fun interaction(completionModel: CompletionModel, config: CompletionConfig) {
        println("Write something")
        var input = "hello wo"
        while (true) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)
            println("$context:$prefix")

            val completions = completionModel.complete(context, prefix, config)
            println(completions)
            println()

            input = readLine().toString()
        }
    }

    private fun speedTest(completionModel: CompletionModel, config: CompletionConfig, len: Int = 1, itersNum: Int = 100) {
//            1 - 0.97258
        println("Warm up")
        val input = (0 until len).map { "hello " }.joinToString { "" } + "wo"
//            val input = "hello wo"
        for (i in 0 until 10) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)

            val completions = completionModel.complete(context, prefix, config)
        }

        println("Start")
        val startTime = currentTimeMillis()
        for (i in 0 until itersNum) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)

            val completions = completionModel.complete(context, prefix, config)
        }
        val duration = (currentTimeMillis() - startTime) / 1000.0 / itersNum
        println()
        println(duration)
    }
}
