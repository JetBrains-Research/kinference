package io.kinference.algorithms.gec.tokenizer.subword.spacy.en

import io.kinference.algorithms.gec.tokenizer.subword.spacy.SpacyTokenInfo

/** TokenizerExceptions class for generating/storing specific exception for tokenizer */
@SuppressWarnings("ComplexMethod", "LongMethod")
class SpacyEnglishTokenizerExceptions {

    /** Words to exclude from exceptions */
    private val exclude = listOf("Ill", "ill", "Its", "its", "Hell",
        "hell", "Shell", "shell", "Shed", "shed", "were", "Were", "Well", "well", "Whore", "whore",)

    /** every possible combination of pronoun/verb and a lot of specific words for right tokenization of this words */
    val exceptions = HashMap<String, List<SpacyTokenInfo>>()

    init {
        generateException()
    }

    private fun generateException(){
        for (pron in listOf("i")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth'm"] = listOf( SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'m", lemma = "be", norm = "am")
                )

                exceptions[orth + "m"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "m", lemma = "be")
                )

                exceptions["$orth'ma"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'m", lemma = "be", norm = "am"),
                    SpacyTokenInfo(orth = "a", lemma = "going to", norm = "gonna")
                )

                exceptions[orth + "ma"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "m", lemma = "be", norm = "am"),
                    SpacyTokenInfo(orth = "a", lemma = "going to", norm = "gonna")
                )

            }
        }

        for (pron in listOf("i", "you", "he", "she", "it", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){

                exceptions["$orth'll"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'ll", lemma = "will", norm = "will")
                )

                exceptions[orth + "ll"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "ll", lemma = "will", norm = "will")
                )

                exceptions["$orth'll've"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'ll", lemma = "will", norm = "will"),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[orth + "llve"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "ll", norm = "will", lemma = "will"),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )

                exceptions["$orth'd"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'d", norm = "'d")
                )

                exceptions[orth + "d"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "d", norm = "'d")
                )

                exceptions["$orth'd've"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'d", lemma = "would", norm = "would"),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[orth + "dve"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "d", lemma = "would", norm = "would"),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )
            }
        }

        for (pron in listOf("i", "you", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth've"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[orth + "ve"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )
            }
        }

        for (pron in listOf("you", "we", "they")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth're"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'re", lemma = "be", norm = "are")
                )

                exceptions[orth + "re"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "re", lemma = "be", norm = "are")
                )
            }
        }

        for (pron in listOf("he", "she", "it")){
            for (orth in listOf(pron, pron.capitalize())){
                exceptions["$orth's"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "'s", norm = "'s")
                )

                exceptions[orth + "s"] = listOf(
                    SpacyTokenInfo(orth = orth, norm = pron),
                    SpacyTokenInfo(orth = "s")
                )
            }
        }

        for (word in listOf("who", "what", "when", "where", "why", "how", "there", "that", "this", "these", "those")){
            for (orth in listOf(word, word.capitalize())){
                exceptions["$orth's"] = listOf(SpacyTokenInfo(orth = orth, lemma = word, norm = word), SpacyTokenInfo(orth = "'s", norm = "'s"))

                exceptions[orth + "s"] = listOf(SpacyTokenInfo(orth = orth, lemma = word, norm = word), SpacyTokenInfo(orth = "s"))

                exceptions["$orth'll"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "'ll", lemma = "will", norm = "will")
                )

                exceptions[orth + "ll"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "ll", lemma = "will", norm = "will")
                )

                exceptions["$orth'll've"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "'ll", lemma = "will", norm = "will"),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[orth + "llve"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "ll", lemma = "will", norm = "will"),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )

                exceptions["$orth're"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "'re", lemma = "be", norm = "are")
                )

                exceptions[orth + "re"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "re", lemma = "be", norm = "are")
                )

                exceptions["$orth've"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "'ve", lemma = "have")
                )

                exceptions[orth + "ve"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )

                exceptions["$orth'd"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                SpacyTokenInfo(orth = "'d", norm = "'d")
                )

                exceptions[orth + "d"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "d", norm = "'d")
                )

                exceptions["$orth'd've"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "'d", lemma = "would", norm = "would"),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[orth + "dve"] = listOf(
                    SpacyTokenInfo(orth = orth, lemma = word, norm = word),
                    SpacyTokenInfo(orth = "d", lemma = "would", norm = "would"),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )
            }
        }

        val verbDatas = listOf(
            SpacyTokenInfo(orth = "ca", lemma = "can", norm = "can"),
            SpacyTokenInfo(orth = "do", lemma = "do", norm = "do"),
            SpacyTokenInfo(orth = "does", lemma = "do", norm = "does"),
            SpacyTokenInfo(orth = "did", lemma = "do", norm = "do"),
            SpacyTokenInfo(orth = "had", lemma = "have", norm = "have"),
            SpacyTokenInfo(orth = "may", norm = "may"),
            SpacyTokenInfo(orth = "need", norm = "need"),
            SpacyTokenInfo(orth = "ought", norm = "ought"),
            SpacyTokenInfo(orth = "sha", lemma = "shall", norm = "shall"),
            SpacyTokenInfo(orth = "wo", lemma = "will", norm = "will")
        )

        val verbs2 = listOf(
            SpacyTokenInfo(orth = "could", norm = "could"),
            SpacyTokenInfo(orth = "might", norm = "might"),
            SpacyTokenInfo(orth = "must", norm = "must"),
            SpacyTokenInfo(orth = "should", norm = "should"),
            SpacyTokenInfo(orth = "would", norm = "would")
        )

        for (verbData in verbDatas + verbs2){
            val verbDataC = verbData.copy()
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "n't"] = listOf(data, SpacyTokenInfo(orth = "n't", lemma = "not", norm = "not"))

                exceptions[data.orth + "nt"] = listOf(data, SpacyTokenInfo(orth = "nt", lemma = "not", norm = "not"))

                exceptions[data.orth + "n't've"] = listOf(data,
                    SpacyTokenInfo(orth = "n't", lemma = "not", norm = "not"),
                    SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")
                )

                exceptions[data.orth + "ntve"] = listOf(data,
                    SpacyTokenInfo(orth = "nt", lemma = "not", norm = "not"),
                    SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")
                )
            }
        }
        for (verbData in verbs2){
            val verbDataC = verbData.copy()
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "'ve"] = listOf(data, SpacyTokenInfo(orth = "'ve", lemma = "have"))

                exceptions[data.orth + "ve"] = listOf(data, SpacyTokenInfo(orth = "ve", lemma = "have"))
            }
        }

        val beVariations = listOf(
            SpacyTokenInfo(orth = "ai", lemma = "be"),
            SpacyTokenInfo(orth = "are", lemma = "be", norm = "are"),
            SpacyTokenInfo(orth = "is", lemma = "be", norm = "are"),
            SpacyTokenInfo(orth = "was", lemma = "be", norm = "is"),
            SpacyTokenInfo(orth = "was", lemma = "be", norm = "was"),
            SpacyTokenInfo(orth = "were", lemma = "be", norm = "were"),
            SpacyTokenInfo(orth = "have", norm = "have"),
            SpacyTokenInfo(orth = "has", lemma = "have", norm = "has"),
            SpacyTokenInfo(orth = "dare", norm = "dare")
        )

        for (verbData in beVariations){
            val verbDataC = verbData.copy()
            verbDataC.orth = verbDataC.orth.capitalize()
            for (data in listOf(verbData, verbDataC)){
                exceptions[data.orth + "n't"] = listOf(data, SpacyTokenInfo(orth = "n't", lemma = "not", norm = "not"))

                exceptions[data.orth + "nt"] = listOf(data, SpacyTokenInfo(orth = "nt", lemma = "not", norm = "not"))
            }
        }

        val trailingApostrpopheData = listOf(
            SpacyTokenInfo(orth = "doin", lemma = "do", norm = "doing"),
            SpacyTokenInfo(orth = "goin", lemma = "go", norm = "going"),
            SpacyTokenInfo(orth = "nothin", lemma = "nothing", norm = "nothing"),
            SpacyTokenInfo(orth = "nuthin", lemma = "nothing", norm = "nothing"),
            SpacyTokenInfo(orth = "ol", lemma = "old", norm = "old"),
            SpacyTokenInfo(orth = "somethin", lemma = "something", norm = "something")
        )
        for (excData in trailingApostrpopheData){
            val excDataC = excData.copy()
            excDataC.orth = excDataC.orth.capitalize()
            for (data in listOf(excData)){
                val dataApos = data
                dataApos.orth = dataApos.orth + "'"
                exceptions[data.orth] = listOf(data)
                exceptions[dataApos.orth] = listOf(dataApos)
            }
        }

        val otherData = listOf(
            SpacyTokenInfo(orth = "em", norm = "them"),
            SpacyTokenInfo(orth = "ll", lemma = "will", norm = "will"),
            SpacyTokenInfo(orth = "nuff", lemma = "enough", norm = "enough")
        )

        for (excData in otherData){
            val excDataApos = excData.copy()
            excDataApos.orth = "'" + excDataApos.orth
            for (data in listOf(excData, excDataApos)){
                exceptions[data.orth] = listOf(data)
            }
        }

        // Times

        for (h in 1..12){
            for (period in listOf("a.m.", "am")){
                exceptions["$h$period"] = listOf(
                    SpacyTokenInfo(orth = "$h"),
                    SpacyTokenInfo(orth = period, lemma = "a.m.", norm = "a.m.")
                )
            }
            for (period in listOf("p.m.", "pm")){
                exceptions["$h$period"] = listOf(
                    SpacyTokenInfo(orth = "$h"),
                    SpacyTokenInfo(orth = period, lemma = "a.m.", norm = "a.m.")
                )
            }
        }

        val otherExc = hashMapOf("y'all" to listOf(SpacyTokenInfo(orth = "y'", norm = "you"), SpacyTokenInfo(orth = "all")),
                                 "yall" to listOf(SpacyTokenInfo(orth = "y", norm = "you"), SpacyTokenInfo(orth = "all")),
                                 "how'd'y" to listOf(SpacyTokenInfo(orth = "how", lemma = "how"), SpacyTokenInfo(orth = "'d", lemma = "do"), SpacyTokenInfo(orth = "'y", norm = "you")),
                                 "How'd'y" to listOf(SpacyTokenInfo(orth = "How", lemma = "how", norm = "how"), SpacyTokenInfo(orth = "'d", lemma = "do"), SpacyTokenInfo(orth = "'y", norm = "you")),
                                 "not've" to listOf(SpacyTokenInfo(orth = "not", lemma = "not"), SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")),
                                 "notve" to listOf(SpacyTokenInfo(orth = "not", lemma = "not"), SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")),
                                 "Not've" to listOf(SpacyTokenInfo(orth = "Not", lemma = "not", norm = "not"), SpacyTokenInfo(orth = "'ve", lemma = "have", norm = "have")),
                                 "Notve" to listOf(SpacyTokenInfo(orth = "Not", lemma = "not", norm = "not"), SpacyTokenInfo(orth = "ve", lemma = "have", norm = "have")),
                                 "cannot" to listOf(SpacyTokenInfo(orth = "can", lemma = "can"), SpacyTokenInfo(orth = "not", lemma = "not")),
                                 "Cannot" to listOf(SpacyTokenInfo(orth = "Can", lemma = "can", norm = "can"), SpacyTokenInfo(orth = "not", lemma = "not")),
                                 "gonna" to listOf(SpacyTokenInfo(orth = "gon", lemma = "go", norm = "going"), SpacyTokenInfo(orth = "na", lemma = "to", norm = "to")),
                                 "Gonna" to listOf(SpacyTokenInfo(orth = "Gon", lemma = "go", norm = "going"), SpacyTokenInfo(orth = "na", lemma = "to", norm = "to")),
                                 "gotta" to listOf(SpacyTokenInfo(orth = "got"), SpacyTokenInfo(orth = "ta", lemma = "to", norm = "to")),
                                 "Gotta" to listOf(SpacyTokenInfo(orth = "Got", norm = "got"), SpacyTokenInfo(orth = "ta", lemma = "to", norm = "to")),
                                 "let's" to listOf(SpacyTokenInfo(orth = "let"), SpacyTokenInfo(orth = "'s", norm = "us")),
                                 "Let's" to listOf(SpacyTokenInfo(orth = "Let", lemma = "let", norm = "let"), SpacyTokenInfo(orth = "'s", norm = "us")),
                                 "c'mon" to listOf(SpacyTokenInfo(orth = "c'm", norm = "come", lemma = "come"), SpacyTokenInfo(orth = "on")),
                                 "C'mon" to listOf(SpacyTokenInfo(orth = "C'm", norm = "come", lemma = "come"), SpacyTokenInfo(orth = "on")))
        exceptions.putAll(otherExc)

        val abbreviations = listOf(
            SpacyTokenInfo(orth = "'S", lemma = "'s", norm = "'s"),
            SpacyTokenInfo(orth = "'s", lemma = "'s", norm = "'s"),
            SpacyTokenInfo(orth = "\\u2018S", lemma = "'S", norm = "'S"),
            SpacyTokenInfo(orth = "\\u2018s", lemma = "'s", norm = "'s"),
            SpacyTokenInfo(orth = "and/or", lemma = "and/or", norm = "and/or"),
            SpacyTokenInfo(orth = "w/o", lemma = "without", norm = "without"),
            SpacyTokenInfo(orth = "'re", lemma = "be", norm = "are"),
            SpacyTokenInfo(orth = "'Cause", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'cause", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'cos", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'Cos", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'coz", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'Coz", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'cuz", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'Cuz", lemma = "because", norm = "because"),
            SpacyTokenInfo(orth = "'bout", lemma = "about", norm = "about"),
            SpacyTokenInfo(orth = "ma'am", lemma = "madam", norm = "madam"),
            SpacyTokenInfo(orth = "Ma'am", lemma = "madam", norm = "madam"),
            SpacyTokenInfo(orth = "o'clock", lemma = "o'clock", norm = "o'clock"),
            SpacyTokenInfo(orth = "O'clock", lemma = "o'clock", norm = "o'clock"),
            SpacyTokenInfo(orth = "lovin'", lemma = "love", norm = "loving"),
            SpacyTokenInfo(orth = "Lovin'", lemma = "love", norm = "loving"),
            SpacyTokenInfo(orth = "lovin", lemma = "love", norm = "loving"),
            SpacyTokenInfo(orth = "Lovin", lemma = "love", norm = "loving"),
            SpacyTokenInfo(orth = "havin'", lemma = "have", norm = "having"),
            SpacyTokenInfo(orth = "Havin'", lemma = "have", norm = "having"),
            SpacyTokenInfo(orth = "havin", lemma = "have", norm = "having"),
            SpacyTokenInfo(orth = "Havin", lemma = "have", norm = "having"),
            SpacyTokenInfo(orth = "doin'", lemma = "do", norm = "doing"),
            SpacyTokenInfo(orth = "Doin'", lemma = "do", norm = "doing"),
            SpacyTokenInfo(orth = "doin", lemma = "do", norm = "doing"),
            SpacyTokenInfo(orth = "Doin", lemma = "do", norm = "doing"),
            SpacyTokenInfo(orth = "goin'", lemma = "go", norm = "going"),
            SpacyTokenInfo(orth = "Goin'", lemma = "go", norm = "going"),
            SpacyTokenInfo(orth = "goin", lemma = "go", norm = "going"),
            SpacyTokenInfo(orth = "Goin", lemma = "go", norm = "going"),
            SpacyTokenInfo(orth = "Mt.", lemma = "Mount", norm = "Mount"),
            SpacyTokenInfo(orth = "Ak.", lemma = "Alaska", norm = "Alaska"),
            SpacyTokenInfo(orth = "Ala.", lemma = "Alabama", norm = "Alabama"),
            SpacyTokenInfo(orth = "Apr.", lemma = "April", norm = "April"),
            SpacyTokenInfo(orth = "Ariz.", lemma = "Arizona", norm = "Arizona"),
            SpacyTokenInfo(orth = "Ark.", lemma = "Arkansas", norm = "Arkansas"),
            SpacyTokenInfo(orth = "Aug.", lemma = "August", norm = "August"),
            SpacyTokenInfo(orth = "Calif.", lemma = "California", norm = "California"),
            SpacyTokenInfo(orth = "Colo.", lemma = "Colorado", norm = "Colorado"),
            SpacyTokenInfo(orth = "Conn.", lemma = "Connecticut", norm = "Connecticut"),
            SpacyTokenInfo(orth = "Dec.", lemma = "December", norm = "December"),
            SpacyTokenInfo(orth = "Del.", lemma = "Delaware", norm = "Delaware"),
            SpacyTokenInfo(orth = "Feb.", lemma = "February", norm = "February"),
            SpacyTokenInfo(orth = "Fla.", lemma = "Florida", norm = "Florida"),
            SpacyTokenInfo(orth = "Ga.", lemma = "Georgia", norm = "Georgia"),
            SpacyTokenInfo(orth = "Ia.", lemma = "Iowa", norm = "Iowa"),
            SpacyTokenInfo(orth = "Id.", lemma = "Idaho", norm = "Idaho"),
            SpacyTokenInfo(orth = "Ill.", lemma = "Illinois", norm = "Illinois"),
            SpacyTokenInfo(orth = "Ind.", lemma = "Indiana", norm = "Indiana"),
            SpacyTokenInfo(orth = "Jan.", lemma = "January", norm = "January"),
            SpacyTokenInfo(orth = "Jul.", lemma = "July", norm = "July"),
            SpacyTokenInfo(orth = "Jun.", lemma = "June", norm = "June"),
            SpacyTokenInfo(orth = "Kan.", lemma = "Kansas", norm = "Kansas"),
            SpacyTokenInfo(orth = "Kans.", lemma = "Kansas", norm = "Kansas"),
            SpacyTokenInfo(orth =  "Ky.", lemma = "Kentucky", norm = "Kentucky"),
            SpacyTokenInfo(orth = "La.", lemma = "Louisiana", norm = "Louisiana"),
            SpacyTokenInfo(orth = "Mar.", lemma = "March", norm = "March"),
            SpacyTokenInfo(orth = "Mass.", lemma = "Massachusetts", norm = "Massachusetts"),
            SpacyTokenInfo(orth = "May.", lemma = "May", norm = "May"),
            SpacyTokenInfo(orth = "Mich.", lemma = "Michigan", norm = "Michigan"),
            SpacyTokenInfo(orth = "Minn.", lemma = "Minnesota", norm = "Minnesota"),
            SpacyTokenInfo(orth = "Miss.", lemma = "Mississippi", norm = "Mississippi"),
            SpacyTokenInfo(orth = "N.C.", lemma = "North Carolina", norm = "North Carolina"),
            SpacyTokenInfo(orth = "N.D.", lemma = "North Dakota", norm = "North Dakota"),
            SpacyTokenInfo(orth = "N.H.", lemma = "New Hampshire", norm = "New Hampshire"),
            SpacyTokenInfo(orth = "N.J.", lemma = "New Jersey", norm = "New Jersey"),
            SpacyTokenInfo(orth = "N.M.", lemma = "New Mexico", norm = "New Mexico"),
            SpacyTokenInfo(orth = "N.Y.", lemma = "New York", norm = "New York"),
            SpacyTokenInfo(orth = "Neb.", lemma = "Nebraska", norm = "Nebraska"),
            SpacyTokenInfo(orth = "Nebr.", lemma = "Nebraska", norm = "Nebraska"),
            SpacyTokenInfo(orth = "Nev.", lemma = "Nevada", norm = "Nevada"),
            SpacyTokenInfo(orth = "Nov.", lemma = "November", norm = "November"),
            SpacyTokenInfo(orth = "Oct.", lemma = "October", norm = "October"),
            SpacyTokenInfo(orth = "Okla.", lemma = "Oklahoma", norm = "Oklahoma"),
            SpacyTokenInfo(orth = "Ore.", lemma = "Oregon", norm = "Oregon"),
            SpacyTokenInfo(orth = "Pa.", lemma = "Pennsylvania", norm = "Pennsylvania"),
            SpacyTokenInfo(orth = "S.C.", lemma = "South Carolina", norm = "South Carolina"),
            SpacyTokenInfo(orth = "Sep.", lemma = "September", norm = "September"),
            SpacyTokenInfo(orth = "Sept.", lemma = "September", norm = "September"),
            SpacyTokenInfo(orth = "Tenn.", lemma = "Tennessee", norm = "Tennessee"),
            SpacyTokenInfo(orth = "Va.", lemma = "Virginia", norm = "Virginia"),
            SpacyTokenInfo(orth = "Wash.", lemma = "Washington", norm = "Washington"),
            SpacyTokenInfo(orth = "Wis.", lemma = "Wisconsin", norm = "Wisconsin")
        )

        for (abb in abbreviations){
            exceptions[abb.orth] = listOf(abb)
        }

        val anotherOneExceptions = listOf("'d",
            "a.m.",
            "Adm.",
            "Bros.",
            "co.",
            "Co.",
            "Corp.",
            "D.C.",
            "Dr.",
            "e.g.",
            "E.g.",
            "E.G.",
            "Gen.",
            "Gov.",
            "i.e.",
            "I.e.",
            "I.E.",
            "Inc.",
            "Jr.",
            "Ltd.",
            "Md.",
            "Messrs.",
            "Mo.",
            "Mont.",
            "Mr.",
            "Mrs.",
            "Ms.",
            "p.m.",
            "Ph.D.",
            "Prof.",
            "Rep.",
            "Rev.",
            "Sen.",
            "St.",
            "vs.",
            "v.s.")

        for (exception in anotherOneExceptions){
            exceptions[exception] = listOf(SpacyTokenInfo(orth = exception))
        }

        for (excludeString in exclude){
            if (excludeString in exceptions.keys){
                exceptions.remove(excludeString)
            }
        }
    }
}
