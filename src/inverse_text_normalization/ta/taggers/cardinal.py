
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pynini
from pynini.lib import pynutil, utf8

from inverse_text_normalization.ta.data_loader_utils import get_abs_path
from inverse_text_normalization.ta.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from inverse_text_normalization.ta.utils import num_to_word
# from inverse_text_normalization.lang_params import LANG
# data_path = f'data/{LANG}_data/'
data_path = 'data/'

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus twenty three -> cardinal { integer: "23" negative: "-" } }
    Numbers below thirteen are not converted. 
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        # integer, negative

        NEMO_CHAR = utf8.VALID_UTF8_CHAR
        NEMO_SIGMA = pynini.closure(NEMO_CHAR)
        NEMO_SPACE = " "
        NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", u"\u00A0").optimize()
        NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
        # NEMO_NON_BREAKING_SPACE = u"\u00A0"

        tamil_digit_file = get_abs_path(data_path + "numbers/ta_digit.tsv")
        with open(tamil_digit_file, encoding='utf-8') as f:
            digits = f.readlines()
        tamil_digits = ''.join([line.split()[-1] for line in digits])
        tamil_digits_with_zero = "0" + tamil_digits

        TAMIL_DIGIT = pynini.union(*tamil_digits).optimize()
        TAMIL_DIGIT_WITH_ZERO = pynini.union(*tamil_digits_with_zero).optimize()

        tamil_graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/ta_zero.tsv"))
        tamil_graph_tens = pynini.string_file(get_abs_path(data_path + "numbers/ta_tens.tsv"))
        tamil_graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_digit.tsv"))
        tamil_graph_hundred_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_hundred_digit.tsv"))
        tamil_graph_thousand_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_thousand_digit.tsv"))
        tamil_graph_lakh_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_lakh_digit.tsv"))
        tamil_graph_crore_digit = pynini.string_file(get_abs_path(data_path + "numbers/ta_crore_digit.tsv"))
        tamil_graph_exception_list = pynini.string_file(get_abs_path(data_path + "numbers/ta_exceptions.tsv"))

        graph_zero = pynini.string_file(get_abs_path(data_path + "numbers/zero.tsv"))  
        graph_digit = pynini.string_file(get_abs_path(data_path + "numbers/digit.tsv"))
        graph_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples.tsv"))
        graph_ties = pynini.string_file(get_abs_path(data_path + "numbers/ties.tsv"))
        graph_chars = pynini.string_file(get_abs_path(data_path + "numbers/alphabets.tsv"))
        graph_char_multiples = pynini.string_file(get_abs_path(data_path + "numbers/multiples_alphabets.tsv"))
        graph_tens_en = pynini.string_file(get_abs_path(data_path + "numbers/tens-en.tsv"))

        tamil_cents = pynini.accep("ற்று") | pynini.accep('த்தி')
        tamil_thousands = pynini.accep('யிரத்து') | pynini.accep('யிரத்தி') | pynini.accep('யிரம்')
        tamil_lakhs = pynini.accep('லட்சம்') | pynini.accep('லட்சத்து')
        tamil_crores = pynini.accep('கோடி') | pynini.accep('கோடியே')

        cents = pynini.accep("ஹண்ட்ரட்‌") | pynini.accep("ஹண்ட்ரெட்‌") | pynini.accep("ஹன்ட்ரட்‌") | pynini.accep("ஹன்ட்ரெட்‌")
        thousands = pynini.accep("தவுசண்ட்‌") | pynini.accep("தௌசண்ட்‌")
        lakhs = pynini.accep("லேக்‌") | pynini.accep("லாக்‌")
        crores = pynini.accep("க்ரோர்‌")

        del_And = pynutil.delete(pynini.closure(pynini.accep("அண்ட்‌"), 1 ,1 ))

        tamil_graph_hundred = pynini.cross("நூறு", "100") | pynini.cross("இருநூறு", "200") | pynini.cross("எரணூறு", "200") |\
                               pynini.cross("முந்நூறு", "300") | pynini.cross("நானூறு", "400") | pynini.cross("ஐநூறு", "500") |\
                               pynini.cross("அறுநூறு", "600") | pynini.cross("எழுநூறு", "700") | pynini.cross("எண்ணூறு","800") | pynini.cross("தொள்ளாயிரம்‌", "900")
        tamil_graph_nine_hundred = pynini.cross("தொள்ளாயிரத்து", "9") | pynini.cross("தொள்ளாயிரத்தி" , "9")
        tamil_graph_thousands = pynini.cross("ஆயிரம்" , "1000")
        tamil_graph_lakhs = pynini.cross("ஒரு லட்சம்", "100000") | pynini.cross("லட்சம்", "100000")
        tamil_graph_crores = pynini.cross("ஒரு கோடி", "10000000") | pynini.cross("கோடி", "10000000")

        graph_hundred = pynini.cross("ஹண்ட்ரட்‌", "100") | pynini.cross("ஹண்ட்ரெட்‌", "100") | pynini.cross("ஹன்ட்ரட்‌", "100") | pynini.cross("ஹன்ட்ரெட்‌", "100")
        graph_thousand  = pynini.cross("தவுசண்ட்‌", "1000") | pynini.cross("தௌசண்ட்‌", "1000")
        graph_lakh = pynini.cross("லேக்‌", "100000") | pynini.cross("லாக்‌", "100000")
        graph_crore = pynini.cross("க்ரோர்‌", "10000000")
        

        #Handles 1-999 (direct spoken)
        tamil_graph_hundred_component = pynini.union(tamil_graph_hundred_digit + pynutil.delete(tamil_cents) + delete_space,
                                                     tamil_graph_nine_hundred + delete_space,
                                                      pynutil.insert("0"))
        tamil_graph_hundred_component += pynini.union(tamil_graph_tens, 
                                                      pynutil.insert("0") + (tamil_graph_digit | pynutil.insert("0")))

        #Handles double digit hundred (like उन्निस सौ) -> Are these present in Tamil??
        ### These variations dont occur in Tamil

        graph_hundred_component = pynini.union((graph_digit | pynutil.insert("1")) + delete_space + pynutil.delete(cents) + (delete_space + del_And + delete_space | delete_space),
                                               pynutil.insert("0"))
        graph_hundred_component += pynini.union((graph_ties  | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))
        # handling double digit hundreds like उन्निस सौ + digit/thousand/lakh/crore etc
        #graph_hundred_component_prefix_tens = pynini.union(graph_tens + delete_space + pynutil.delete(cents) + delete_space,)
        #                                                   # pynutil.insert("55"))
        graph_hundred_component_prefix_tens = pynini.union((graph_tens_en) + delete_space + pynutil.delete(cents) + (delete_space + del_And + delete_space | delete_space),
                                                            )

        graph_hundred_component_prefix_tens += pynini.union((graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))

        # #Handles 1-99
        # tamil_graph_hundred_component_non_hundred = pynini.union(tamil_graph_tens,
        #                                                          pynutil.insert("0") + (tamil_graph_digit | pynutil.insert("0")))

        # tamil_graph_hundred_component = pynini.union(tamil_graph_hundred_component, )

        #Handles 10-99 in both hi, en
        graph_hundred_component_non_hundred = pynini.union((graph_ties | pynutil.insert("0")) + delete_space + (graph_digit | pynutil.insert("0")))

        #This thing now handles only 100-999 cases (in regular spoken form) and 1000-9999 (in hundred spoken form)
        #Because of combining these both FSTs, there comes ambiguity while dealing with 1-99 cases.
        graph_hundred_component = pynini.union(graph_hundred_component,
                                               graph_hundred_component_prefix_tens)

        graph_hundred_component_at_least_one_none_zero_digit = pynini.union(graph_hundred_component, graph_hundred_component_non_hundred)



        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )


        # self.graph_hundred_component_at_least_one_none_zero_digit = (
        #     tamil_graph_hundred_component
        # )

        tamil_graph_thousands_component = pynini.union(tamil_graph_thousand_digit + pynutil.delete(tamil_thousands),
                                                       pynutil.insert("00", weight=-0.1))
        
        tamil_graph_lakhs_component = pynini.union(tamil_graph_lakh_digit + delete_space + pynutil.delete(tamil_lakhs),
                                                   pynutil.insert("00", weight=-0.1))
        
        tamil_graph_crore_component = pynini.union(tamil_graph_crore_digit + delete_space + pynutil.delete(tamil_crores),
                                               pynutil.insert("00", weight=-0.1))


        graph_thousands_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(thousands),
            (pynutil.insert("0") + graph_hundred_component_prefix_tens),
            pynutil.insert("00", weight=-0.1),
        )

        graph_lakhs_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(lakhs),
            pynutil.insert("00", weight=-0.1)
        )

        graph_crores_component = pynini.union(
            (graph_hundred_component_at_least_one_none_zero_digit + delete_space | pynutil.insert("1", weight=-0.1)) + pynutil.delete(crores),
            pynutil.insert("00", weight=-0.1)
        )

        
        fst_tam = pynini.union(
            tamil_graph_crore_component
            + (delete_space)
            + tamil_graph_lakhs_component
            + (delete_space)
            + (tamil_graph_thousands_component)
            + (delete_space)
            + (tamil_graph_hundred_component | pynutil.insert("", weight=0.1)),
            graph_zero,
        )

        fst_en = pynini.union(
            graph_crores_component
            + (delete_space | delete_space + del_And + delete_space)
            + graph_lakhs_component
            + (delete_space | delete_space + del_And + delete_space)
            + (graph_thousands_component)
            + (delete_space | delete_space + del_And + delete_space)
            + (graph_hundred_component | pynutil.insert("", weight=0.1)),
        )
        fst_crore = fst_en+graph_crore # handles words like चार हज़ार करोड़
        fst_lakh = fst_en+graph_lakh # handles words like चार हज़ार लाख

        fst = pynini.union(fst_tam, tamil_graph_crores , tamil_graph_lakhs, tamil_graph_thousands, tamil_graph_hundred, tamil_graph_exception_list,
                           fst_en, fst_crore, fst_lakh, graph_crore, graph_lakh, graph_thousand, graph_hundred, graph_multiples)

        self.graph_no_exception = fst
        self.graph = fst

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

                                                      
