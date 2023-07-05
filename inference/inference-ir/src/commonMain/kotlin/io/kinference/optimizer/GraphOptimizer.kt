package io.kinference.optimizer

import io.kinference.data.ONNXData
import io.kinference.graph.Graph
import io.kinference.operator.Operator
import io.kinference.utils.LoggerFactory

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
    if (!visited.contains(op.name)) visited.add(op.name)
    if (targetOpTypes[currentOp] == op.type) {
        path.add(OperatorLocation(op, startIdx))
    }
    else
        return false

    val successors = operators.successorsOf(startIdx)
    if (successors.isEmpty() && this.outputs.map { it.name }.containsAll(op.outputs)) return true

    for (successor in successors) {
        if (!visited.contains(successor.operator.name)) {
            if (this.findPathRec(targetOpTypes, successor.idx, currentOp + 1, visited, path))
                return true
        }
    }
    path.removeLast()
    return false
}

fun <T : ONNXData<*, *>> Graph<T>.findPath(targetOpTypes: List<String>, startIdx: Int): List<Operator<T, T>>? {
    if (operators[startIdx].type != targetOpTypes[0]) return null

    val path = ArrayList<OperatorLocation<T>>()

    val visited = ArrayList<String>()
    val found = this.findPathRec(targetOpTypes, startIdx, 0, visited, path)
    return if (found) path.map { it.operator } else null
}

class GraphOptimizer<T : ONNXData<*, *>>(val graph: Graph<T>) {
    class OptimizationReport {
        private val report = HashMap<String, Int>()

        fun append(rule: OptimizerRule<*>) {
            if (rule.name in report.keys)
                report[rule.name] = report[rule.name]!! + 1
            else
                report[rule.name] = 1
        }

        override fun toString(): String {
            val strReport = StringBuilder().appendLine("Number of applied graph transformations:")
            for ((name, counts) in report.entries) {
                strReport.appendLine("$name: $counts")
            }
            return strReport.toString()
        }
    }

    suspend fun run(rules: List<OptimizerRule<T>>): Graph<T> {
        val report = OptimizationReport()
        for (rule in rules) rule.apply(graph, report)

        logger.info { report.toString() }

        return graph
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.optimizer.GraphOptimizer")

        fun optName(name: String?) = "${OptimizerRule.PREFIX}_${name!!}"
    }
}
