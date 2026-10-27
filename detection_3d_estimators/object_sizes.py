# Copyright 2026 Rodrigo Pérez-Rodríguez
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

"""The real size of what a detection is, from its label: a closed class of a detector (YOLO,
COCO: "person", "cup") or an open-vocabulary label ("person with a red shirt", "the red coffee
cup on the table"). No ROS here.

The label is matched to a name of the table (or one of its aliases) by its head noun: the words
before the first preposition or relative word ("with", "on", "in", "who", ...) are the noun
phrase, and its last word that names something in the table is the head ("red coffee cup" ->
cup; "person with a cup" -> person). Names of several words ("dining table", "cell phone") are
matched as a whole. If nothing in the noun phrase matches, the rest of the label is searched.
"""

import re
from dataclasses import dataclass

import yaml

# Words that end the noun phrase of a label: what comes after describes the thing, it is not it
STOP_WORDS = {
    'with', 'without', 'on', 'in', 'at', 'near', 'next', 'beside', 'behind', 'under', 'above',
    'over', 'of', 'from', 'to', 'for', 'by', 'holding', 'wearing', 'carrying', 'sitting',
    'standing', 'walking', 'who', 'that', 'which', 'and', 'or', 'left', 'right', 'closest',
    'farthest', 'nearest',
}
ARTICLES = {'a', 'an', 'the', 'this', 'my', 'your', 'his', 'her', 'their', 'its'}


@dataclass(frozen=True)
class Size:
    name: str
    height: float = 0.0   # m; 0: unknown
    width: float = 0.0    # m; 0: unknown


def words(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def singular(word):
    """A plain singular, enough for the names of the table (cups -> cup, glasses stays)."""
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


class ObjectSizes:
    """The table of sizes: name -> Size, with the aliases of each name."""

    def __init__(self, table):
        self.sizes = {}    # name -> Size
        self.names = {}    # tuple of words of a name or alias -> name
        for name, entry in (table or {}).items():
            entry = entry or {}
            self.sizes[name] = Size(name, float(entry.get('height') or 0.0),
                                    float(entry.get('width') or 0.0))
            for alias in [name] + list(entry.get('aliases') or []):
                self.names[tuple(singular(w) for w in words(str(alias)))] = name
        self.longest = max((len(k) for k in self.names), default=1)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def override(self, specs):
        """Sizes given as 'name:height:width' (width optional; a new name is added)."""
        for spec in specs or []:
            parts = [p.strip() for p in str(spec).split(':')]
            if len(parts) < 2 or not parts[0]:
                raise ValueError(f'A size is "name:height[:width]", not "{spec}"')
            name = parts[0].lower()
            height = float(parts[1] or 0.0)
            width = float(parts[2]) if len(parts) > 2 and parts[2] else (
                self.sizes[name].width if name in self.sizes else 0.0)
            self.sizes[name] = Size(name, height, width)
            self.names[tuple(singular(w) for w in words(name))] = name
            self.longest = max(self.longest, len(words(name)))

    def restrict(self, names):
        """Keep only these names (and their aliases); none: keep all."""
        if not names:
            return
        keep = {n.lower() for n in names}
        unknown = keep - set(self.sizes)
        if unknown:
            raise ValueError(f'No size for {", ".join(sorted(unknown))}: add them with `sizes`')
        self.sizes = {n: s for n, s in self.sizes.items() if n in keep}
        self.names = {k: n for k, n in self.names.items() if n in keep}

    def _matches(self, tokens):
        """The names found in a list of words: (position of its last word, number of words,
        name), longest names first where they overlap."""
        found = []
        for start in range(len(tokens)):
            for length in range(min(self.longest, len(tokens) - start), 0, -1):
                name = self.names.get(tuple(tokens[start:start + length]))
                if name:
                    found.append((start + length - 1, length, name))
                    break
        return found

    def match(self, label):
        """The Size of what the label names, or None."""
        tokens = [singular(w) for w in words(label) if w not in ARTICLES]
        if not tokens:
            return None
        end = next((i for i, w in enumerate(tokens) if w in STOP_WORDS and i > 0), len(tokens))
        for part in (tokens[:end], tokens[end:]):
            found = self._matches(part)
            if found:
                # The head of the noun phrase: the last name in it (a longer one where they end
                # at the same word: "dining table" over "table")
                _, _, name = max(found)
                return self.sizes[name]
        return None
