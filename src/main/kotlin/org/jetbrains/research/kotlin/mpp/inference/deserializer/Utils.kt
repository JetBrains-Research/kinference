package org.jetbrains.research.kotlin.mpp.inference.deserializer

import scientifik.kmath.linear.VirtualMatrix
import scientifik.kmath.linear.transpose
import scientifik.kmath.structures.Matrix

internal inline fun <reified T : Number> List<T>.toDoubleList(): List<Double> = map { it.toDouble() }

internal fun FloatArray.toDoubleList() = asList().toDoubleList()
internal fun Array<FloatArray>.toDoubleList() = map { it.toDoubleList() }

@Suppress("UNCHECKED_CAST")
internal inline fun <reified T> T.toMatrix(): Matrix<*> = when (this) {
    is FloatArray -> VirtualMatrix(rowNum = size, colNum = 1, generator = { i, _ -> this[i] }).transpose()
    is Array<*> -> {
        this as Array<FloatArray>
        VirtualMatrix(rowNum = size, colNum = this[0].size, generator = { i, j -> this[i][j] }).transpose()
    }
    else -> throw IllegalStateException("Cannot cast ${T::class} to matrix")
}
