package io.kinference.ndarray

class ConvParameters(
    val pads: Any,
    val strides: IntArray,
    val dilations: IntArray,
    val groups: Int,
    val rank: Int) {
    data class Builder(
        private var dimensions: Int? = null,
        private var autoPad: AutoPad = AutoPad.NOTSET,
        private var pads: IntArray? = null,
        private var strides: IntArray? = null,
        private var dilations: IntArray? = null,
        private var groups: Int = 1
    ) {
        fun specifyDimensions(dimensions: Int) = apply { this.dimensions = dimensions }

        fun specifyPads(pads: IntArray?) = apply { this.pads = pads }

        fun specifyAutoPad(autoPad: String) = apply { this.autoPad = AutoPad.valueOf(autoPad) }

        fun specifyStrides(strides: IntArray?) = apply { this.strides = strides }

        fun specifyDilations(dilations: IntArray?) = apply { this.dilations = dilations }

        fun specifyGroups(groups: Int) = apply { this.groups = groups }

        private fun ones(size: Int) = IntArray(size) { 1 }

        private fun defaultStrides() = ones(dimensions!!)

        private fun defaultDilations() = ones(dimensions!!)

        private fun inferPads(): Any {
            if (pads != null) {
                // Ensure the pads array has the correct number of elements
                require(pads!!.size == 2 * dimensions!!) { "\"pads\" array must have ${2 * dimensions!!} elements for ${dimensions!!}D convolution." }

                val tfjsPads = Array(dimensions!! + 2) { IntArray(2) { 0 } }
                // ONNX pads array is in the order of [begin0, begin1, ..., end0, end1, ...]
                for (i in 0 until dimensions!!) {
                    tfjsPads[i + 2 - dimensions!! % 2][0] = pads!![i] // Begin padding
                    tfjsPads[i + 2 - dimensions!! % 2][1] = pads!![i + dimensions!!] // End padding
                }
                return tfjsPads
            } else {
                return autoPad.tfjsValue
            }
        }

        enum class AutoPad(val tfjsValue: String) {
            NOTSET("valid"),
            VALID("valid"),
            SAME_UPPER("same"),
            SAME_LOWER("same")
        }

        fun build(): ConvParameters {
            require(dimensions != null) { "Number of dimensions must be specified before build." }

            if (strides == null)
                strides = defaultStrides()

            if (dilations == null)
                dilations = defaultDilations()

            return ConvParameters(inferPads(), strides!!, dilations!!, groups, dimensions!!)
        }
    }
}
