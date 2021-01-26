package io.kinference.algorithms.gec.pretokenizer.en

import io.kinference.algorithms.gec.pretokenizer.CharClasses

/**
 * Prefix class which contains information about prefixes for English text
 * @param prefixes list of Strings which contains punctuation, ellipses, quotes, currencies, icons and several strings for prefix matching
 * @param prefixesRegex Regex string which allows to search prefix in text
 */
class Prefix {
    private val prefixes =
        listOf("§", "%", "=", "—", "–", """\+(?![0-9])""") +
            CharClasses.ListPunct +
            CharClasses.ListEllipses +
            CharClasses.ListQuotes +
            CharClasses.ListCurrency +
            CharClasses.ListIcons

    val prefixesRegex: Regex?

    init {
        prefixesRegex = compileRegex()
    }

    private fun compileRegex(): Regex?{
        val expr: Regex
        if (prefixes.isNotEmpty()){
            expr = if ("(" in prefixes){
                prefixes.filter { it.trim() != "" }.joinToString("|") { "^" + Regex.escape(it) }.toRegex()
            } else{
                prefixes.filter { it.trim() != "" }.joinToString("|") { "^$it" }.toRegex()
            }
            return expr
        }
        return null
    }
}
