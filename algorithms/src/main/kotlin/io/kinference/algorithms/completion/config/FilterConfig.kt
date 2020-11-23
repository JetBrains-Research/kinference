package io.kinference.algorithms.completion.config

data class FilterConfig(val minSymbolLen: Int, val minAvgLogProb: Double, val minProb: Double) {
    companion object {
        val default = FilterConfig(
            minSymbolLen = 2,
            minAvgLogProb = -100.0,
            minProb = 0.0
        )
    }
}
