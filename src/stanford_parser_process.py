"""Stanford ParserのJavaプログラムをダウンロードして、サーバモードとして起動するモジュール

基本的にこのモジュールを直接利用する必要はなく、stanford_parserモジュールを利用するほうが良い
"""
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
from nltk.parse.corenlp import CoreNLPServer


class StanfordParserProcess(CoreNLPServer):
    """Stanford ParserのJavaプログラムをダウンロードして、サーバモードとして起動するクラス

    `~/.cache/stanford_parser`にStanford Parserをインストールして、
    サーバモードとして起動する。

    >>> from nltk.parse.corenlp import CoreNLPParser
    >>> proc = StanfordParserProcess()
    >>> parser = CoreNLPParser(proc.get_endpoint())
    >>> with proc:
    ...     # you can use a stanford parser process as HTTP Server
    ...     sentences = list(parser.parse(parser.tokenize("This is a pen.")))
    ...     tree = sentences[0]
    ...     print(tree)
    """

    CACHE_DIR = os.path.expanduser("~/.cache/stanford_parser")
    PARSER_DIR = os.path.expanduser("~/.cache/stanford_parser/stanford_parser")
    INSTALL_PATHS = {
        "3.9.1": {
            "files": [
                {
                    "url": "http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip",
                    "dest": "stanford-corenlp-full-2018-02-27",
                    "extract": "zip",
                },
            ],
            "route_link": "stanford-corenlp-full-2018-02-27",
        },
        "4.5.5": {
            "files": [
                {
                    "url": "https://nlp.stanford.edu/software/stanford-corenlp-4.5.5.zip",
                    "dest": "stanford-corenlp-4.5.5.zip",
                    "extract": "zip",
                },
                {
                    "url": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-english.jar",
                    "dest": "stanford-corenlp-4.5.5/stanford-corenlp-4.4.0-models-english.jar",
                    "extract": None,
                },
            ],
            "route_link": "stanford-corenlp-4.5.5",
        },
    }

    def __init__(self, version="4.5.5", port=None):
        """Stanford parserを起動"""
        self._started = False

        self.install_corenlp(version)
        os.environ["CLASSPATH"] = self.PARSER_DIR + ":" + (os.environ["CLASSPATH"] if "CLASSPATH" in os.environ.keys() else "")
        super().__init__(port=port)

    def get_started(self):
        return self._started

    def get_endpoint(self):
        return self.url

    def __enter__(self):
        obj = super().__enter__()
        self._started = True
        return obj

    def __exit__(self, *args):
        obj = super().__exit__(*args)
        self._started = False
        return obj

    @staticmethod
    def download(url, path_):
        """urlで指定されたファイルをpathに保存する"""
        file_path = Path(path_)

        if file_path.exists():
            return False

        os.makedirs(file_path.parent, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=file_path.parent) as temp_dir:
            temp_path = os.path.join(temp_dir, file_path.name)
            req = requests.get(url, stream=True)
            full_size = int(req.headers.get("content-length"))

            # output to temporary file
            with open(temp_path, "wb") as file:
                with tqdm(total=full_size, unit='iB', unit_scale=True, desc=f"Download {file_path.name}") as bar:
                    for chunk in req.iter_content(chunk_size=8192):
                        if not chunk:
                            break
                        file.write(chunk)
                        bar.update(len(chunk))

            # if failed to downloda, raise Error
            req.raise_for_status()

            # move to destination
            os.replace(temp_path, file_path)

        return True

    @staticmethod
    def unzip(path):
        """pathで指定されたzipファイルを展開する"""
        with zipfile.ZipFile(path) as zfile:
            zfile.extractall(Path(path).parent)

    @classmethod
    def install_corenlp(cls, version, clear_cache=False):
        available_versions = cls.INSTALL_PATHS.keys()
        if version not in available_versions:
            print(
                f"error: unkown version of corenlp for {version}. You must specify from {available_versions}",
                file=sys.stderr,
            )
            return False

        install_config = cls.INSTALL_PATHS[version]

        for file in install_config["files"]:
            if Path(file["dest"]).exists():
                break

            dest_path = os.path.join(cls.CACHE_DIR, file["dest"])
            if not file["extract"]:
                cls.download(file["url"], dest_path)
            elif file["extract"].lower() == "zip":
                zip_dest_path = os.path.join(cls.CACHE_DIR, file["dest"] + ".zip")
                cls.download(file["url"], zip_dest_path)
                cls.unzip(zip_dest_path)
                if clear_cache:
                    os.remove(zip_dest_path)

        current_ver = Path(os.path.join(cls.CACHE_DIR, install_config["route_link"]))
        symlink = Path(cls.PARSER_DIR)
        if symlink.exists() and symlink.is_symlink() and symlink.absolute() == current_ver.absolute():
            return True

        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(current_ver)
