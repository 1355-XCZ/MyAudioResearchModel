# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.utils import list2str, str2list
from typing import List, Union
import os
import json
import sys

# Windows espeak修复 - 设置DLL库路径
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll')
    print("✅ 设置espeak DLL路径成功")
except Exception as e:
    print(f"⚠️ espeak DLL路径设置失败: {e}")

# separator=Separator(phone=' ', word=' _ ', syllable='|'),
separator = Separator(word=" _ ", syllable="|", phone=" ")

try:
    phonemizer_zh = EspeakBackend(
        "cmn", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
    )
except RuntimeError:
    # Windows espeak路径修复
    import os
    os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG\espeak-ng.exe'
    phonemizer_zh = EspeakBackend(
        "cmn", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
    )
# phonemizer_zh.separator = separator

try:
    phonemizer_en = EspeakBackend(
        "en-us",
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
    )
except RuntimeError:
    # Windows espeak路径修复
    import os
    os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG\espeak-ng.exe'
    phonemizer_en = EspeakBackend(
        "en-us",
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
    )
# phonemizer_en.separator = separator

try:
    phonemizer_ja = EspeakBackend(
        "ja", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
    )
except RuntimeError as e:
    print(f"⚠️ 日语语音包不可用: {e}")
    phonemizer_ja = None
# phonemizer_ja.separator = separator

try:
    phonemizer_ko = EspeakBackend(
        "ko", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
    )
except RuntimeError as e:
    print(f"⚠️ 韩语语音包不可用: {e}")
    phonemizer_ko = None
# phonemizer_ko.separator = separator

try:
    phonemizer_fr = EspeakBackend(
        "fr-fr",
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
    )
except RuntimeError as e:
    print(f"⚠️ 法语语音包不可用: {e}")
    phonemizer_fr = None
# phonemizer_fr.separator = separator

try:
    phonemizer_de = EspeakBackend(
        "de", preserve_punctuation=False, with_stress=False, language_switch="remove-flags"
    )
except RuntimeError as e:
    print(f"⚠️ 德语语音包不可用: {e}")
    phonemizer_de = None
# phonemizer_de.separator = separator


lang2backend = {
    "zh": phonemizer_zh,
    "ja": phonemizer_ja,
    "en": phonemizer_en,
    "fr": phonemizer_fr,
    "ko": phonemizer_ko,
    "de": phonemizer_de,
}

import pathlib
current_dir = pathlib.Path(__file__).parent
mls_en_path = str(current_dir / "mls_en.json")
with open(mls_en_path, "r", encoding='utf-8') as f:
    json_data = f.read()
token = json.loads(json_data)


def phonemizer_g2p(text, language):
    langbackend = lang2backend[language]
    phonemes = _phonemize(
        langbackend,
        text,
        separator,
        strip=True,
        njobs=1,
        prepend_text=False,
        preserve_empty_lines=False,
    )
    token_id = []
    if isinstance(phonemes, list):
        for phone in phonemes:
            phonemes_split = phone.split(" ")
            token_id.append([token[p] for p in phonemes_split if p in token])
    else:
        phonemes_split = phonemes.split(" ")
        token_id = [token[p] for p in phonemes_split if p in token]
    return phonemes, token_id


def _phonemize(  # pylint: disable=too-many-arguments
    backend,
    text: Union[str, List[str]],
    separator: Separator,
    strip: bool,
    njobs: int,
    prepend_text: bool,
    preserve_empty_lines: bool,
):
    """Auxiliary function to phonemize()

    Does the phonemization and returns the phonemized text. Raises a
    RuntimeError on error.

    """
    # remember the text type for output (either list or string)
    text_type = type(text)

    # force the text as a list
    text = [line.strip(os.linesep) for line in str2list(text)]

    # if preserving empty lines, note the index of each empty line
    if preserve_empty_lines:
        empty_lines = [n for n, line in enumerate(text) if not line.strip()]

    # ignore empty lines
    text = [line for line in text if line.strip()]

    if text:
        # phonemize the text
        phonemized = backend.phonemize(
            text, separator=separator, strip=strip, njobs=njobs
        )
    else:
        phonemized = []

    # if preserving empty lines, reinsert them into text and phonemized lists
    if preserve_empty_lines:
        for i in empty_lines:  # noqa
            if prepend_text:
                text.insert(i, "")
            phonemized.insert(i, "")

    # at that point, the phonemized text is a list of str. Format it as
    # expected by the parameters
    if prepend_text:
        return list(zip(text, phonemized))
    if text_type == str:
        return list2str(phonemized)
    return phonemized
