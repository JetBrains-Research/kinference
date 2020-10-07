package io.kinference.completion

import io.kinference.completion.suggest.*

class MainCompletion {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val config = config

            val completionsCollector = FairseqCompletionsCollector(config)
            val rankingModel = FirstProbRankingModel()
            val filterModel = ProbFilterModel()
            val postFilterModel: FilterModel? = null

            val completionModel = CompletionModel(completionsCollector, rankingModel, filterModel, postFilterModel)

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
    }
}
