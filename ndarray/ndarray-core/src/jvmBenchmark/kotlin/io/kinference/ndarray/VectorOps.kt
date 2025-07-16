package io.kinference.ndarray

import jdk.incubator.vector.*


abstract class OperationNode() {
    abstract fun vecResolve(idx: Int): FloatVector
    abstract fun linResolve(idx: Int): Float
    final fun resolveTo(dest: FloatArray, offset: Int, len: Int) {
        val end = len - (len % vecSize)
        for (idx in 0 until end step vecSize) {
            vecResolve(idx).intoArray(dest, offset + idx)
        }
        for (idx in end until len) dest[offset + idx] = linResolve(idx)
    }

    companion object {
        val species = FloatVector.SPECIES_PREFERRED
        val vecSize = species.length()
    }

    abstract fun genVecCode(nodeId: String): String
    abstract fun genLinCode(nodeId: String): String
    fun genCode(): String {
        return """
            val end = len - (len % $vecSize)
            for (idx in 0 until end step $vecSize) {
                ${genVecCode("root")}
                .intoArray(dest, destOffset + idx)
            }
            for (idx in end until len) {
                dest[destOffset + idx] = ${genLinCode("root")}
            } 
        """.trimIndent()

    }

    fun genReduce(op: String): String {
        return """
            val acc = FloatVector.zero(species)
            val end = len - (len % $vecSize)
            for (idx in 0 until end step $vecSize) {
                ${genVecCode("root")}
            }
            for (idx in end until len) {
            
            }
        """
    }
}

class FloatSlice(val src: FloatArray, val offset: Int) : OperationNode() {
    override fun linResolve(idx: Int): Float = src[offset + idx]
    override fun vecResolve(idx: Int): FloatVector = FloatVector.fromArray(species, src, offset + idx)
    override fun genVecCode(nodeId: String): String {
        return "FloatVector.fromArray(species, src$nodeId, offset$nodeId + idx)"
    }

    override fun genLinCode(nodeId: String): String {
        return "src$nodeId[offset$nodeId + idx]"
    }

}

class Constant(val value: Float) : OperationNode() {
    override fun linResolve(idx: Int): Float = value
    override fun genVecCode(nodeId: String): String {
        return "FloatVector.broadcast(species, $value)"
    }

    override fun genLinCode(nodeId: String): String {
        return "$value"
    }

    override fun vecResolve(idx: Int): FloatVector = FloatVector.broadcast(species, value)
}

abstract class UnaryOperation(val arg: OperationNode) : OperationNode() {
    abstract fun apply(arg: Float): Float
    abstract val vectorHandle: VectorOperators.Unary
    final override fun linResolve(idx: Int): Float = apply(arg.linResolve(idx))
    final override fun vecResolve(idx: Int): FloatVector =
        arg.vecResolve(idx).lanewise(vectorHandle)

    final override fun genVecCode(nodeId: String): String {
        return "${arg.genVecCode(nodeId + "l")}\n.lanewise(${vectorHandle.name()})"
    }

}

abstract class BinaryOperation(val first: OperationNode, val second: OperationNode) : OperationNode() {
    abstract fun apply(first: Float, second: Float): Float
    abstract val vectorHandle: VectorOperators.Binary
    final override fun linResolve(idx: Int): Float = apply(first.linResolve(idx), second.linResolve(idx))
    final override fun vecResolve(idx: Int): FloatVector =
        first.vecResolve(idx).lanewise(vectorHandle, second.vecResolve(idx))

    final override fun genVecCode(nodeId: String): String {
        return "${second.genVecCode(nodeId + "r")}\n.lanewise(${vectorHandle.name()}, ${first.genVecCode(nodeId + "l")})"
    }
}

abstract class AssociativeOperation(first: OperationNode, second: OperationNode) : BinaryOperation(first, second) {

}

//abstract class TernaryOperation(val first: OperationNode, val second: OperationNode, val third: OperationNode) : OperationNode() {
//    abstract fun apply(first: Float, second: Float, third: Float): Float
//    abstract val vectorHandle: String
//}


class Add(first: OperationNode, second: OperationNode) : AssociativeOperation(first, second) {

    override fun apply(first: Float, second: Float): Float {
        return first + second
    }

    final override val vectorHandle = VectorOperators.ADD
    override fun genLinCode(nodeId: String): String {
        return "(${first.genLinCode(nodeId + "l")} + ${second.genLinCode(nodeId + "r")})"
    }
}

class Sub(first: OperationNode, second: OperationNode) : BinaryOperation(first, second) {

    override fun apply(first: Float, second: Float): Float {
        return first - second
    }

    override fun genLinCode(nodeId: String): String {
        return "(${first.genLinCode(nodeId + "l")} - ${second.genLinCode(nodeId + "r")})"
    }

    final override val vectorHandle = VectorOperators.SUB
}

class Mul(first: OperationNode, second: OperationNode) : AssociativeOperation(first, second) {
    override fun apply(first: Float, second: Float): Float {
        return first * second
    }

    override fun genLinCode(nodeId: String): String {
        return "(${first.genLinCode(nodeId + "l")} * ${second.genLinCode(nodeId + "r")})"
    }

    final override val vectorHandle: VectorOperators.Associative = VectorOperators.MUL
}

class Div(first: OperationNode, second: OperationNode) : BinaryOperation(first, second) {
    final override fun apply(first: Float, second: Float): Float {
        return first * second
    }

    override fun genLinCode(nodeId: String): String {
        return "(${first.genLinCode(nodeId + "l")} / ${second.genLinCode(nodeId + "r")})"
    }

    final override val vectorHandle = VectorOperators.DIV
}

class Max(first: OperationNode, second: OperationNode) : AssociativeOperation(first, second) {
    final override fun apply(first: Float, second: Float): Float {
        return first
    }

    override fun genLinCode(nodeId: String): String {
        return "maxOf(${first.genLinCode(nodeId + "l")},  ${second.genLinCode(nodeId + "r")})"
    }

    final override val vectorHandle = VectorOperators.MAX
}

class Exp(first: OperationNode) : UnaryOperation(first) {
    final override fun apply(arg: Float): Float {
        return kotlin.math.exp(arg)
    }

    override fun genLinCode(nodeId: String): String {
        return "exp(${arg.genLinCode(nodeId + "l")})"
    }

    final override val vectorHandle = VectorOperators.EXP
}

fun main() {
    val a = FloatArray(0)
    val b = FloatArray(0)
    val c = FloatArray(0)

    val op = Add(FloatSlice(a, 0), Constant(69F))
    println(op.genCode())

}
