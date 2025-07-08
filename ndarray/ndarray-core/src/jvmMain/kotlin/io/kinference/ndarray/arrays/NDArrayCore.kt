package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.*

interface NDArrayCore : NDArray {
    override suspend fun clone(): NDArrayCore
    override fun close() = Unit

    suspend fun map(function: PrimitiveToPrimitiveFunction, destination: MutableNDArray): MutableNDArrayCore
    suspend fun map(function: PrimitiveToPrimitiveFunction): MutableNDArrayCore

    override fun view(vararg axes: Int): NDArrayCore

    override suspend fun reshape(strides: Strides): NDArrayCore
    override suspend fun reshape(shape: IntArray): NDArrayCore = reshape(Strides(shape))

    override suspend fun toMutable(): MutableNDArrayCore

    override suspend fun transpose(permutations: IntArray): NDArrayCore

    override suspend fun squeeze(vararg axes: Int): NDArrayCore = squeeze(this, *axes)
    override suspend fun unsqueeze(vararg axes: Int): NDArray = unsqueeze(this, *axes)

    override suspend fun stack(others: List<NDArray>, axis: Int): MutableNDArrayCore = (listOf(this) + others as List<NDArrayCore>).stack(axis)
    override suspend fun concat(others: List<NDArray>, axis: Int): MutableNDArrayCore

    override suspend fun split(parts: Int, axis: Int): List<NDArrayCore> = this.splitWithAxis(parts, axis, true)
    override suspend fun split(split: IntArray, axis: Int): List<NDArray> = this.splitWithAxis(split, axis, true)

    override suspend fun gather(indices: NDArray, axis: Int, batchDims: Int): NDArrayCore = gather(this, indices as NDArrayCore, axis)
    suspend fun gather(indices: NDArray, axis: Int, dst: MutableNDArrayCore): NDArrayCore = gather(this, indices as NDArrayCore, axis, dst)

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArrayCore
}

interface MutableNDArrayCore : NDArrayCore, MutableNDArray {
    suspend fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArrayCore
    override fun viewMutable(vararg axes: Int): MutableNDArrayCore
}

interface NumberNDArrayCore : NDArrayCore, NumberNDArray {
    suspend fun gemm(
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        lda: Int,
        b: NDArray,
        ldb: Int,
        beta: Double,
        c: MutableNDArray,
        ldc: Int,
        aOffset: Int,
        bOffset: Int,
        cOffset: Int,
        transposeA: Boolean,
        transposeB: Boolean
    ): MutableNDArrayCore

    override suspend fun toMutable(): MutableNumberNDArrayCore

    override suspend fun reshape(strides: Strides): NumberNDArrayCore
    override suspend fun reshape(shape: IntArray): NumberNDArrayCore = reshape(Strides(shape))

    override fun view(vararg axes: Int): NumberNDArrayCore
    override suspend fun transpose(permutations: IntArray): NumberNDArrayCore

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArrayCore

    suspend fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayCore
    suspend fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayCore
    suspend fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayCore
    suspend fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayCore

    override suspend operator fun plus(other: NumberNDArray): MutableNumberNDArrayCore
    override suspend operator fun minus(other: NumberNDArray): MutableNumberNDArrayCore
    override suspend operator fun times(other: NumberNDArray): MutableNumberNDArrayCore
    override suspend operator fun div(other: NumberNDArray): MutableNumberNDArrayCore

    override suspend fun softmax(axis: Int): NumberNDArrayCore
    override suspend fun logSoftmax(axis: Int): NumberNDArrayCore

    suspend fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArrayCore
    suspend fun matmul(other: NumberNDArray, destination: MutableNumberNDArrayCore): MutableNumberNDArrayCore
}

interface MutableNumberNDArrayCore : NumberNDArrayCore, MutableNDArrayCore, MutableNumberNDArray {
    override fun viewMutable(vararg axes: Int): MutableNumberNDArrayCore
}
