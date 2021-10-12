package io.kinference.profiler

import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

data class ProfileEntry(val name: String, val time: Long, val children: List<ProfileEntry>) {
    fun writeToStringBuilder(builder: StringBuilder, indent: Int = 0) {
        builder.append("${" ".repeat(indent)}$name - ${time / 1_000_000f} ms\n")
        for (entry in children) entry.writeToStringBuilder(builder, indent + 1)
    }
}

data class ProfileAnalysisEntry(
    val name: String,

    val avg: Double,
    val max: Double,
    val min: Double,
    val p50: Double,
    val p90: Double,

    val children: List<ProfileAnalysisEntry>
) {
    private fun Double.toRoundString(): String {
        val rounded = this.toInt()
        return "${rounded}.${((this - rounded) * 100.0).roundToInt().toString().padStart(2, '0')}".padStart(7)
    }

    fun writeToStringBuilder(builder: StringBuilder, indent: Int = 0) {
        builder.append("\n${("  ".repeat(indent) + name).padEnd(40)} AVG: ${avg.toRoundString()} ms | MAX: ${max.toRoundString()} ms | MIN: ${min.toRoundString()} ms | Percentile 50: ${p50.toRoundString()} ms | Percentile 90: ${p90.toRoundString()} ms")
        for (entry in children) {
            entry.writeToStringBuilder(builder, indent + 1)
        }
    }

    fun getInfo(): String = buildString { writeToStringBuilder(this) }
}

internal fun ProfileEntry.writeToAnalysisBuilder(builder: ProfileAnalysisBuilder) {
    // TODO: support conditional/loop operators
    require(this.name == builder.name) { "Entries must be from the same profiling context" }

    builder.times.add(this.time)

    if (builder.children.size == 0 && children.isNotEmpty()) {
        for (child in children) {
            builder.children.add(ProfileAnalysisBuilder(child.name))
        }
    }

    require(children.size == builder.children.size) { "Entries must be from the same profiling context" }

    for ((index, child) in children.withIndex()) {
        child.writeToAnalysisBuilder(builder.children[index])
    }
}

internal data class ProfileAnalysisBuilder(val name: String,
                                          val times: MutableList<Long> = ArrayList(),
                                          val children: MutableList<ProfileAnalysisBuilder> = ArrayList()) {
    fun analyze(): ProfileAnalysisEntry {
        require(times.isNotEmpty()) { "Nothing to analyze" }

        val analyzedChildren: MutableList<ProfileAnalysisEntry> = ArrayList()
        for (child in children) {
            analyzedChildren.add(child.analyze())
        }

        times.sort()
        return ProfileAnalysisEntry(
            name,
            times.average() / 1_000_000.0,
            times.last() / 1_000_000.0,
            times.first() / 1_000_000.0,
            times[floor(0.5 * times.size).toInt()] / 1_000_000.0,
            times[floor(0.9 * times.size).toInt()] / 1_000_000.0,
            analyzedChildren
        )
    }
}
