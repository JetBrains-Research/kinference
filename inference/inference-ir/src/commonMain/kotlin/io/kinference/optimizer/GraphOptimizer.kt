package io.kinference.optimizer

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.Operator

fun <T : ONNXData<*, *>> List<Operator<T, T>>.predecessorsOf(idx: Int): List<OperatorLocation<T>> {
    val op = this[idx]
    val predecessors = ArrayList<OperatorLocation<T>>()
    for (i in idx - 1 downTo 0) {
        if (this[i].outputs.intersect(op.inputs).isNotEmpty())
            predecessors.add(OperatorLocation(this[i], i))
    }
    return predecessors
}

fun <T : ONNXData<*, *>> List<Operator<T, T>>.successorsOf(idx: Int): List<OperatorLocation<T>> {
    val op = this[idx]
    val successors = ArrayList<OperatorLocation<T>>()
    for (i in idx + 1 until this.size) {
        if (this[i].inputs.intersect(op.outputs).isNotEmpty())
            successors.add(OperatorLocation(this[i], i))
    }
    return successors
}

data class OperatorLocation<T : ONNXData<*, *>>(val operator: Operator<T, T>, val idx: Int)

private fun <T : ONNXData<*, *>> Graph<T>.findPathRec(targetOpTypes: List<String>, startIdx: Int, currentOp: Int, visited: ArrayList<String>, path: ArrayList<OperatorLocation<T>>): Boolean {
    if (currentOp == targetOpTypes.size && path.size == targetOpTypes.size) return true

    val op = operators[startIdx]
    if (!visited.contains(op.name)) visited.add(operators[startIdx].name)
    if (targetOpTypes[currentOp] == operators[startIdx].type)
        path.add(OperatorLocation(operators[startIdx], startIdx))
    else
        return false

    val successors = operators.successorsOf(startIdx)
    for (i in successors.indices) {
        if (!visited.contains(successors[i].operator.name)) {
            if (this.findPathRec(targetOpTypes, successors[i].idx, currentOp + 1, visited, path)) {
                return true
            }
        }
    }
    path.removeLast()
    return false
}

fun <T : ONNXData<*, *>> Graph<T>.findPath(targetOpTypes: List<String>, startIdx: Int): List<OperatorLocation<T>>? {
    if (operators[startIdx].type != targetOpTypes[0]) return null

    val path = ArrayList<OperatorLocation<T>>()

    val visited = ArrayList<String>()
    val found = this.findPathRec(targetOpTypes, startIdx, 0, visited, path)
    return if (found) path else null
}

class GraphOptimizer<T : ONNXData<*, *>>(val graph: Graph<T>) {
    fun run(rules: Set<OptimizerRule<T>>): Graph<T> {
        for (rule in rules) rule.apply(graph)

        return graph
    }
}
