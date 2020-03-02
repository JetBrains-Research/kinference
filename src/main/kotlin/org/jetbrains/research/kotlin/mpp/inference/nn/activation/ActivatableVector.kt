package org.jetbrains.research.kotlin.mpp.inference.nn.activation

import scientifik.kmath.linear.*
import scientifik.kmath.structures.Buffer

class ActivatableVector(size: Int) {
    var values: RealMatrix = BufferMatrix(size, 1, Buffer.auto(size) { 0.0 })
        private set

    private var notActivatedValues: RealMatrix? = null

    fun activate(func: ActivationFunction) {
        if (notActivatedValues == null) notActivatedValues = values
        values = func(notActivatedValues!!.asPoint())
    }

    fun forward(weights: RealMatrix, bias: RealMatrix?, x: RealMatrix): RealMatrix {
        var res = weights.dot(x)
        if (bias != null) res = MatrixContext.real.add(res, bias.transpose())
        notActivatedValues = res
        return res
    }
}
