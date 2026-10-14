# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Select functionals for additional tuning variants without removing base images."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class TuningFilter:
    kernel: str
    arch: str
    choices: dict[str, list]

    @classmethod
    def from_file(cls, path):
        with open(path) as stream:
            selector = cls(**json.load(stream))
        assert isinstance(selector.kernel, str) and selector.kernel
        assert isinstance(selector.arch, str) and selector.arch
        assert isinstance(selector.choices, dict)
        assert all(isinstance(values, list) and values
                   for values in selector.choices.values())
        return selector

    def matches(self, functional):
        if (functional.arch != self.arch
                or not functional.meta_object.unique_path.match(self.kernel)):
            return False
        return all(getattr(functional.choices, name) in values
                   for name, values in self.choices.items())
