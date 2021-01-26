package io.kinference.algorithms.gec.tokenizer.subword.spacy

import io.kinference.algorithms.gec.tokenizer.subword.spacy.en.SpacyEnglish
import io.kinference.algorithms.gec.tokenizer.subword.spacy.en.SpacyEnglishTokenizerExceptions

/**
 * SpecialCases is class-container for exceptions and url searching string
 * @param baseExceptions basic exceptions for tokenizer
 * @param tokenizerExceptions tokenizer exceptions for tokenizer
 * @param specialCases hash map for matching special token and information about this token
 * @param urls string which generates url-matching regex
 * @param urlRegex regex which match urls
 */
class SpacyTokenizerSpecialCases {
    private val tokenizerExceptions = SpacyEnglishTokenizerExceptions()

    private val specialCases: MutableMap<String, List<SpacyTokenInfo>> = (tokenizerExceptions.exceptions + SpacyEnglish.BaseExceptions.exceptions).toMutableMap()


    private val urls = """(?u)""" +
        // fmt: off
        """^""" +
        // protocol identifier (mods: make optional and expand schemes)
        // (see: https://www.iana.org/assignments/uri-schemes/uri-schemes.xhtml)
        """(?:(?:[\w\+\-\.]{2,})://)?""" +
        // mailto:user or user:pass authentication
        """(?:\S+(?::\S*)?@)?""" +
        """(?:""" +
        // IP address exclusion
        // private & local networks
        """(?!(?:10|127)(?:\.\d{1,3}){3})""" +
        """(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})""" +
        """(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})""" +
        // IP address dotted notation octets
        // excludes loopback network 0.0.0.0
        // excludes reserved space >= 224.0.0.0
        // excludes network & broadcast addresses
        // (first & last IP address of each class)
        // MH: Do we really need this? Seems excessive, and seems to have caused
        // Issue #957
        """(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])""" +
        """(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}""" +
        """(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))""" +
        """|""" +
        // host & domain names
        // mods: match is case-sensitive, so include [A-Z]
        """(?:""" +  // noqa
        """(?:""" +
        """[A-Za-z0-9\u00a1-\uffff]""" +
        """[A-Za-z0-9\u00a1-\uffff_-]{0,62}""" +
        """)?""" +
        """[A-Za-z0-9\u00a1-\uffff]\.""" +
        """)+""" +
        // TLD identifier
        // mods: use ALPHA_LOWER instead of a wider range so that this doesn't match
        // strings like "lower.Upper", which can be split on "." by infixes in some
        // languages
        """(?:[""" + SpacyTokenizerCharClasses.AlphaLower + """]{2,63})""" +
        """)""" +
        // port number
        """(?::\d{2,5})?""" +
        // resource path
        """(?:[/?#]\S*)?""" +
        """$"""
    // fmt: on

    private var urlRegex: Regex = urls.toRegex()


    init {
        expandExceptions(search = "'", replace = "’")
    }


    private fun expandExceptions(search: String, replace: String) {
        fun fixToken(token: SpacyTokenInfo, search: String, replace: String): SpacyTokenInfo {
            return SpacyTokenInfo(orth = token.orth.replace(search, replace), norm = token.norm, tag = token.tag, lemma = token.lemma)
        }

        val newItems = HashMap<String, List<SpacyTokenInfo>>()
        for ((tokenString, tokens) in specialCases) {
            if (search in tokenString) {
                val newKey = tokenString.replace(search, replace)
                val newValue = tokens.map { fixToken(it, search, replace) }
                newItems[newKey] = newValue
            }
        }
        specialCases.putAll(newItems)
    }

    fun get(token: String): List<SpacyTokenInfo>? {
        return if (specialCases.containsKey(token)) {
            specialCases[token]
        } else {
            null
        }
    }

    fun urlMatch(token: String): Boolean {
        val matches = urlRegex.find(token)
        if (matches != null) {
            return true
        }
        return false
    }
}
