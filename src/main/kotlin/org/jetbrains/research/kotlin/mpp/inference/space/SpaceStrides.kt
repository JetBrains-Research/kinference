package org.jetbrains.research.kotlin.mpp.inference.space

import scientifik.kmath.structures.Strides

class SpaceStrides private constructor(override val shape: IntArray) : Strides {
    override val strides by lazy {
        sequence {
            var current = 1
            yield(1)
            shape.drop(1).reversed().forEach {
                current *= it
                yield(current)
            }
        }.toList().reversed()
    }

    override fun offset(index: IntArray): Int {
        return index.mapIndexed { i, value ->
            if (value < 0 || value >= this.shape[i]) {
                throw RuntimeException("Index $value out of shape bounds: (0,${this.shape[i]})")
            }
            value * strides[i]
        }.sum()
    }

    override fun index(offset: Int): IntArray {
        require(offset in 0 until linearSize) { "Offset $offset out of buffer bounds: (0, ${linearSize - 1})" }

        val res = IntArray(shape.size)
        var current = offset

        for (index in strides.indices){
            res[index] = current / strides[index]
            current %= strides[index]
        }

        return res
    }

    override val linearSize: Int
        get() = strides[0] * shape[0]


    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SpaceStrides) return false

        if (!shape.contentEquals(other.shape)) return false

        return true
    }

    override fun hashCode(): Int {
        return shape.contentHashCode()
    }

    companion object {
        private val spaceStridesCache = HashMap<IntArray, Strides>()

        operator fun invoke(shape: IntArray): Strides = spaceStridesCache.getOrPut(shape) { SpaceStrides(shape) }
    }
}
