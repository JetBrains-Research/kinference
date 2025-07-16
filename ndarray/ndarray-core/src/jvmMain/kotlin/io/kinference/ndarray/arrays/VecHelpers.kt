package io.kinference.ndarray.arrays

import jdk.incubator.vector.*

val floatSpecies = FloatVector.SPECIES_PREFERRED
val floatVecSize = floatSpecies.length()

class BinaryVecOp(val src: FloatArray, val offset: Int, val op: VectorOperators.Binary) {
}

// creates a VectorMask with a suffix of setBits ones
inline fun createMask(setBits: Int): VectorMask<Float> {
    return VectorMask<Float>.fromLong(floatSpecies, ((1L shl setBits) - 1L) xor ((1L shl floatVecSize) - 1L))
}

inline fun applyInPlace(src: FloatArray, srcOffset: Int, other: BinaryVecOp, len: Int) {
    val end = len - (len % floatVecSize)
    for (idx in 0 until end step floatVecSize) {
        FloatVector.fromArray(floatSpecies, src, srcOffset + idx)
            .lanewise(other.op, FloatVector.fromArray(floatSpecies, other.src, other.offset + idx))
            .intoArray(src, srcOffset + idx)
    }
    for(idx in end until len) {
        src[srcOffset + idx] += other.src[other.offset + idx]
    }
    //FloatVector.fromArray(floatSpecies, src, srcOffset + len - floatVecSize)
    //    .lanewise(other.op, FloatVector.fromArray(floatSpecies, other.src, other.offset + len - floatVecSize), createMask(len - end))
    //    .intoArray(src, srcOffset + len - floatVecSize)
}

inline fun applyInPlace(src: FloatArray, srcOffset: Int, op: VectorOperators.Binary, other: Float, len: Int) {
    val end = len - (len % floatVecSize)
    for (idx in 0 until end step floatVecSize) {
        FloatVector.fromArray(floatSpecies, src, srcOffset + idx)
            .lanewise(op, other)
            .intoArray(src, srcOffset + idx)
    }
    for(idx in end until len) {
        src[srcOffset + idx] += other
    }
    //FloatVector.fromArray(floatSpecies, src, srcOffset + len - floatVecSize)
    //    .lanewise(op, other, createMask(len - end))
    //    .intoArray(src, srcOffset + len - floatVecSize)
}

inline fun reduce(src: FloatArray, srcOffset: Int, op: VectorOperators.Associative, len: Int): Float {
    var accumulator = FloatVector.fromArray(floatSpecies, src, srcOffset)
    val end = len - (len % floatVecSize)
    for (idx in srcOffset + floatVecSize until end step floatVecSize) {
        accumulator = accumulator.lanewise(op, FloatVector.fromArray(floatSpecies, src, srcOffset + idx))
    }
    accumulator.lanewise(op, FloatVector.fromArray(floatSpecies, src, end - floatVecSize), createMask(len - end))
    return accumulator.reduceLanes(op)
}
