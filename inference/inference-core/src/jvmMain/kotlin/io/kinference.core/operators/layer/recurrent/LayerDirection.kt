package io.kinference.core.operators.layer.recurrent

enum class LayerDirection {
    FORWARD,
    REVERSE,
    BIDIRECTIONAL;

    fun numDirections() = when(this) {
        FORWARD, REVERSE -> 1
        BIDIRECTIONAL -> 2
    }
}
