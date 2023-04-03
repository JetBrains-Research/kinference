package io.kinference.profiler

import io.kinference.utils.time.Duration
import io.kinference.utils.time.Timer


class ProfilingContext(val name: String) {
    private val entries: MutableList<ProfileEntry> = ArrayList()
    private var time: Duration = Duration.ZERO

    suspend fun profile(name: String, block: suspend (context: ProfilingContext?) -> Unit) {
        val context = ProfilingContext(name)
        val time = Timer.measure {
            block(context)
        }

        this.time = this.time + time
        entries.add(ProfileEntry(name, time, context.entries))
    }

    fun entry(): ProfileEntry = ProfileEntry(name, time, entries.toMutableList())
}


suspend fun ProfilingContext?.profile(name: String, block: suspend (context: ProfilingContext?) -> Unit) =
    this?.profile(name, block) ?: block(null)


fun Collection<ProfilingContext>.analyze(name: String): ProfileAnalysisEntry {
    val builder = ProfileAnalysisBuilder(name)
    for (context in this) {
        val profile = context.entry()
        profile.writeToAnalysisBuilder(builder)
    }

    return builder.analyze()
}
