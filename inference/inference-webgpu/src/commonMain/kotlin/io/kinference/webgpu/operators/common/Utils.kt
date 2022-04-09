package io.kinference.webgpu.operators.common

fun shapeToWorkSize(shape: IntArray): IntArray =
    shape.reversedArray().let {
        intArrayOf(it.getOrElse(0) { 1 }, it.getOrElse(1) { 1 }, it.drop(2).fold(1, Int::times))
    }
