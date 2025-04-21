"""stanford parserを利用するためのモジュール

基本的に下記のように、`get_stanford_parser_proc`関数から
得られるオブジェクトをwith構文を利用して起動する。

>>> with get_stanford_parser_proc():
...     main()

ただし、パーサの関数構成などについては最適な構成へのリファクタリングは実施せず、
後方互換性を保つことに重点が置かれている。
"""
import sys
import warnings

import nltk
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser

from stanford_parser_process import StanfordParserProcess


_parser = None
_parser_dep = None
_parser_proc = None


def get_stanford_dependency_parser():
    global _parser_proc, _parser_dep

    if not _parser_proc or not _parser_proc.get_started():
        print("before getting parser, you must start parser process"
              "(by calling get_stanford_parser_proc() as with-context).",
              file=sys.stderr)
        return None

    if not _parser_dep:
        _parser_dep = CoreNLPDependencyParser(url=_parser_proc.get_endpoint())
    return _parser_dep


def get_stanford_parser():
    global _parser_proc, _parser

    if not _parser_proc or not _parser_proc.get_started():
        print("before getting parser, you must start parser process"
              "(by calling get_stanford_parser_proc() as with-context).",
              file=sys.stderr)
        return None

    if not _parser:
        _parser = CoreNLPParser(url=_parser_proc.get_endpoint())
    return _parser


def get_stanford_parser_proc():
    global _parser_proc
    if not _parser_proc:
        _parser_proc = StanfordParserProcess()
    return _parser_proc


def POSTagAnalysis(text):
    """POSタグの分析"""
    # POSタグの分析用
    parser = get_stanford_parser()

    # POSタグの分析(iterator形式で返ってくる)
    out = list(parser.parse(parser.tokenize(text)))

    # Treeを取得 ※テキストは一つと仮定，増えるとout[1]などに格納されるかも
    tree = out[0]

    return tree


def dependenceAnalysis(text):
    """係り受け関係の分析"""
    # 係り受け関係分析 (iterator形式で返ってくる)
    parser_dep = get_stanford_dependency_parser()
    out = list(parser_dep.parse(text))

    # parseを取得 ※テキストは一つと仮定，増えるとout[1]などに格納されるかも
    parse = out[0]

    return parse


def getNodes(parent, np_phrases, pp_phrases):
    """名詞句と前置詞句を再帰的に探索する"""
    previous_np = ""
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == "ROOT":
                print("======== Sentence =========")
                print("Sentence:", " ".join(node.leaves()))
            else:
                if node.label() == "NP":
                    np_phrases.add(" ".join(node.leaves()))
                    previous_np = " ".join(node.leaves())
                if (node.label() == "PP" or node.label() == "ADVP") and previous_np != "":
                    pp_phrases.add(previous_np + " " + " ".join(node.leaves()))

            getNodes(node, np_phrases, pp_phrases)
    return np_phrases, pp_phrases


def getVerbPhrases(parent, vp_phrases):
    """動詞句を再帰的に探索する"""
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == "ROOT":
                print("======== Sentence =========")
                print("Sentence:", " ".join(node.leaves()))
            else:
                if node.label() == "VP":
                    vp_phrases.add(" ".join(node.leaves()))
            getVerbPhrases(node, vp_phrases)
    return vp_phrases


def get_all_np(instruction):
    """すべての名詞句(np)を取得する"""
    tree = POSTagAnalysis(instruction)
    np_phrases, pp_phrases = getNodes(tree, set(), set())
    return np_phrases | pp_phrases


def get_all_vp(instruction):
    """すべての動詞句(vp)を取得する"""
    tree = POSTagAnalysis(instruction)
    vp_phrases = getVerbPhrases(tree, set())
    return vp_phrases


def get_longest_np(instruction):
    """最も文字数の多い名詞句を取得する"""
    all_np = get_all_np(instruction)
    return max(all_np, key=len)


def main():
    instructions = [
        "This is a pen.",
        "When I was a child, I lived in California for 4 years, and that experience made me love spending time in nature, especially national parks like Yosemite.",
    ]

    for inst in instructions:
        # POSタグの分析（上記で定義した関数）
        print("=================")
        print(inst)

        print("-----------------")
        tree = POSTagAnalysis(inst)
        print(tree)
        print()

        print("-----------------")
        print(dependenceAnalysis(inst))
        print()

        print("-----------------")
        print("--NP--")
        for phrase in get_all_np(inst):
            print(phrase)
        print("--VP--")
        for phrase in get_all_vp(inst):
            print(phrase)
        print()


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    with get_stanford_parser_proc():
        main()
