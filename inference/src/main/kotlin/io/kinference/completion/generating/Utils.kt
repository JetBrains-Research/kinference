package io.kinference.completion.generating

fun logSoftmax(scores: List<MutableList<Double>>): List<MutableList<Double>> {
    val expScores = scores.map { it.map { e -> kotlin.math.exp(e) }.toMutableList() }
    val sumLastScores = expScores.map { it.sum() }
    return expScores.mapIndexed { i, list -> list.map { it / sumLastScores[i] }.toMutableList() }
}

fun topk1d(data: List<Double>, size: Int): List<Int> {
    val pairedData = ArrayList<Pair<Double, Int>>()
    for (i in data.indices) {
        pairedData.add(Pair(data[i], i))
    }
    pairedData.sortBy { -it.first }
    return pairedData.map { it.second }.subList(0, size)
}

fun topk2d(data: List<List<Double>>, size: Int, dim: Int = 0): List<List<Int>> {
    return listOf()
//    val pairedData = ArrayList<Pair<Double, Int>>()
//    for (i in data.indices) {
//        pairedData.add(Pair(data[i], i))
//    }
//    pairedData.sortBy { -it.first }
//    return pairedData.map { it.second }.subList(0, size)
}
