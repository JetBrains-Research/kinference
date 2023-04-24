package io.kinference.ndarray.extensions.utils

data class InputInfo(
    val inputShape: IntArray,
    val kernelShape: IntArray,
    val pads: IntArray,
    val strides: IntArray,
    val dilations: IntArray,
    val groups: Int,
    val rank: Int,
    val ceilMode: Int
    ) {
    val outputShape: IntArray by lazy {
        IntArray(rank) {
            val inputWithPad = inputShape[it] + padTotal(it)
            val kernelWithDilation = (kernelShape[it] - 1) * dilations[it] + 1

            if (ceilMode == 1)
                ((inputWithPad - kernelWithDilation) divCeil strides[it]) + 1
            else
                ((inputWithPad - kernelWithDilation) / strides[it]) + 1
        }
    }

    val inputSize: Int by lazy { inputShape.inferShapeSize() }

    val kernelSize: Int by lazy { kernelShape.inferShapeSize() }

    val outputSize: Int by lazy { outputShape.inferShapeSize() }

    fun padBegin(i: Int): Int {
        return pads[i]
    }

    fun padEnd(i: Int): Int {
        return pads[i + rank]
    }

    fun padTotal(i: Int): Int {
        return padBegin(i) + padEnd(i)
    }

    data class Builder(
        private var dimensions: Int? = null,
        private var autoPad: String = "NOTSET",
        private var pads: IntArray? = null,
        private var strides: IntArray? = null,
        private var dilations: IntArray? = null,
        private var inputShape: IntArray? = null,
        private var kernelShape: IntArray? = null,
        private var groups: Int = 1,
        private var ceilMode: Int = 1
    ) {
        fun specifyDimensions(dimensions: Int) = apply { this.dimensions = dimensions }

        fun specifyPads(pads: IntArray?) = apply { this.pads = pads }

        fun specifyInputShape(inputShape: IntArray) = apply { this.inputShape = inputShape }

        fun specifyKernelShape(kernelShape: IntArray) = apply { this.kernelShape = kernelShape }

        fun specifyAutoPad(autoPad: String) = apply { this.autoPad = autoPad }
        
        fun specifyStrides(strides: IntArray?) = apply { this.strides = strides }

        fun specifyDilations(dilations: IntArray?) = apply { this.dilations = dilations }

        fun specifyGroups(groups: Int) = apply { this.groups = groups }

        fun specifyCeilMode(ceilMode: Int) = apply { this.ceilMode = ceilMode }

        private fun ones(size: Int) = IntArray(size) { 1 }

        private fun defaultStrides() = ones(dimensions!!)

        private fun defaultDilations() = ones(dimensions!!)

        private fun defaultPads() = IntArray(dimensions!! * 2) { 0 }

        private fun inferPads() : IntArray {
            require(autoPad == "NOTSET" || pads == null) { "Explicit pads cannot be used simultaneously with auto_pad attribute." }

            if (autoPad == "VALID" || autoPad == "NOTSET")
                return defaultPads()

            require(inputShape != null && strides != null && kernelShape != null && dilations != null) { "inputShape, strides, dilations and kernelShape must be specified, when auto_pad is SAME_UPPER or SAME_LOWER" }
            val pads = IntArray(dimensions!! * 2) {
                if (it < dimensions!!) {
                    val outputShape = inputShape!![it] divCeil strides!![it]
                    (outputShape - 1) * strides!![it] + ((kernelShape!![it] - 1) * dilations!![it] + 1) - inputShape!![it]
                    //(outputShape - 1) * strides!![it] + kernelShape!![it] - inputShape!![it]
                } else {
                    0
                }
            }

            if (autoPad == "SAME_UPPER") {
                for (i in 0 until dimensions!!) {
                    val padEnd = pads[i] divCeil 2
                    pads[i + dimensions!!] = padEnd
                    pads[i] -= padEnd
                }

                return pads
            }

            if (autoPad == "SAME_LOWER") {
                for (i in 0 until dimensions!!) {
                    val padEnd = pads[i] / 2
                    pads[i + dimensions!!] = padEnd
                    pads[i] -= padEnd
                }

                return pads
            }

            error("Invalid auto_pad argument: $autoPad.")
        }

        fun build(): InputInfo {
            require(dimensions != null && kernelShape != null && inputShape != null) { "Number of dimensions, kernelShape, inputShape must be specified before build." }

            if (strides == null)
                strides = defaultStrides()

            if (dilations == null)
                dilations = defaultDilations()

            if (pads == null)
                pads = inferPads()

            return InputInfo(inputShape!!, kernelShape!!, pads!!, strides!!, dilations!!, groups, dimensions!!, ceilMode)
        }
    }
}
